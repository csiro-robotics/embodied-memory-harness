"""MACLA arm: procedural memory after MACLA (arXiv 2512.18950), on the unified harness.

DEC-12: where the MACLA paper and its released code differ, this follows the paper. The
released `MACLA.py` never runs ALFWorld and never calls its own segmentation, refinement
or meta-procedure pipeline. From it we use only the data structures (`Procedure`,
`ProceduralMemoryEntry`, `MetaProcedure`, `ContrastiveContext`), the memory container and
the §4.5 ontology builder. Everything else is written here, section by section:

§4.1 LLM-based procedural abstraction (DEC-11). After each training episode one
     memory-side LLM call segments the trajectory into sub-tasks, each with a goal G,
     preconditions Ψ and postconditions Φ (Eq. 1). From a failed episode only verified
     substeps are kept (§1, §B.1). The action sequence π of a segment is
     the segment's own actions, cleaned and written with placeholders (<object>,
     <target>, <source>; §C.6). Each procedure is embedded as [G; Ψ; Φ] and merged into
     a stored one when the cosine similarity exceeds θ_dup = 0.85 (P15), only among
     procedures with the same key verb. A merge expands the condition sets.
§4.2 Bayesian selection. Each step the observation (with the task) is embedded, the top-k
     procedures and meta-procedures by similarity are scored by Eq. 4, and the best one
     is used if its EU is at least θ_conf; otherwise the step is zero-shot (no memory
     text). θ_conf follows §D.1.2, Eq. 13, with the zero-shot success rate measured on
     this arm's own training episodes (P16).
§4.3 Contrastive refinement. A procedure with at least 3 successes and 3 failures gets an
     LLM comparison of the two sets; its output edits Ψ, π and Φ, which are all shown to
     the agent (P5, P17).
§4.4 Meta-procedures. A sequence of procedures that recurs in at least θ_meta = 15% of the
     last 15 episodes of its task family (App. C: "3/15 = 20%") becomes a meta-procedure
     with its own Beta posterior. It runs under the control policy {continue, skip,
     repeat, abort}, decided from what this episode's observations show (P-meta).
§4.6 Pruning by Eq. 8 when the store is full, stratified by key verb.

Ours, not the paper's (disclosed, P4, P9): the effect-verified cleaning of π, the
placeholders' binding to the task, postcondition checks that read the simulator's
acknowledgements, the learned "object held" precondition, the task-family grouping of
meta-procedures, the abort rule, and the likely-locations list.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

try:
    import MACLA
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("the macla arm needs the MACLA authors' MACLA.py in src/ (see README.md)") from e
from MACLA import ContrastiveContext, LLMMACLAAgent, MetaProcedure, Procedure
from alfworld_env import classify_task_family, extract_target_objects
from memory.base import EpisodeResult, MemoryContext, MemoryModule

log = logging.getLogger(__name__)

# ---- paper constants (§5.1, App. D Table 8) ------------------------------------------
THETA_DUP = 0.85          # duplicate threshold
LAMBDA_INFO = 0.1         # information-gain weight
C_FAIL = 0.5              # failure cost
R_MAX = 1.0
N_MIN = 3                 # refinement needs >= 3 successes and >= 3 failures
META_WINDOW = 15          # "last 15 episodes" (App. C)
THETA_META = 0.15         # a sequence in >= 15% of recent episodes forms a meta-procedure
C_LLM = 0.15              # normalised zero-shot cost in §D.1.2 Eq. 13
TOP_K = 5                 # "top-5 goal-relevant procedures" (App. C)
FAIL_CAP = 15             # failure/success context lists, as in the engine

# P11: MiniLM pinned to one revision (the one cached locally and on M3), so similarities
# near θ_dup cannot move between replicates because a new revision was downloaded.
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# "<noun> <instance>" argument of an action or acknowledgment, e.g. "apple 1".
_ARG_RE = re.compile(r"\b([a-z]+) (\d+)\b")
# Verbs whose acknowledgment certifies a sub-task (vs. navigation / container handling).
_SIGNIFICANT = ("take", "move", "put", "heat", "cool", "clean", "use", "slice")
_PLACE = ("move", "put")

# Fallback acknowledgment patterns, used only until the arm has learned its own
# templates from training observations (unit tests, cold start).
_FALLBACK_ACK = {
    "take": r"you (?:pick up|take) the (?P<a1>[a-z]+) (?P<n1>\d+)"
            r"(?: from the (?P<a2>[a-z]+) (?P<n2>\d+))?",
    "move": r"you (?:move|put) the (?P<a1>[a-z]+) (?P<n1>\d+) "
            r"(?:to|in/on|in|on) the (?P<a2>[a-z]+) (?P<n2>\d+)",
    "go":   r"you arrive at (?P<a1>[a-z]+) (?P<n1>\d+)",
    "use":  r"you turn on the (?P<a1>[a-z]+) (?P<n1>\d+)",
}

SEGMENT_SYSTEM = ("You analyse a finished episode of a text-based household game and split "
                  "it into its sub-tasks.")
SEGMENT_USER = """Below is an episode that {outcome}: the task, the first observation, then each
step's action and the game's reply.

Split the episode into its meaningful sub-tasks, in order. For each sub-task give:
- "start" and "end": the first and last step numbers it covers;
- "goal": what the sub-task achieves, in a few words;
- "preconditions": what was true before it started, as short phrases;
- "postconditions": what was true after it, as short phrases, read from the game's replies.
Steps that only searched without finding anything may be folded into the next sub-task.
Output ONLY a JSON array, for example:
[{{"start": <n>, "end": <n>, "goal": "<goal>", "preconditions": ["<phrase>"], "postconditions": ["<phrase>"]}}]

TASK: {task}
FIRST OBSERVATION: {obs0}
{steps}"""

REFINE_SYSTEM = ("You compare successful and failed uses of one procedure in a text-based "
                 "household game.")
REFINE_USER = """PROCEDURE
goal: {goal}
preconditions: {pre}
steps: {steps}
postconditions: {post}

SUCCESSFUL USES (first observation, then the actions taken):
{succ}

FAILED USES:
{fail}

Find what distinguishes the successful uses from the failed ones:
- "preconditions_positive": conditions present when it succeeded and absent when it failed;
- "preconditions_negative": conditions present when it failed;
- "steps": the procedure's steps, corrected if the successful uses show missing or
  misordered actions (keep the placeholders written in angle brackets); [] if no change;
- "postconditions": results that mark the procedure as complete.
Output ONLY a JSON object with these four keys, each a list of short strings."""


@dataclass
class _Step:
    """One effective action of a trajectory."""
    action: str                      # lowercased raw action
    verb: str
    args: list                       # [(noun, num), ...]
    acked: bool                      # environment acknowledged the effect
    last: bool                       # final action of the episode
    idx: int = 0                     # index in the trajectory


@dataclass
class _Bind:
    """What the task names, resolved against the arm's learned vocabularies."""
    obj: str | None = None
    target: str | None = None
    receptacles: set = field(default_factory=set)   # receptacle nouns in this room


@dataclass
class _State:
    """World facts certified by this episode's own observations."""
    held: tuple | None = None
    taken: set = field(default_factory=set)             # instances ever picked up
    placed: dict = field(default_factory=dict)          # (noun, num) -> receptacle noun
    treated: dict = field(default_factory=lambda: defaultdict(set))
    used: set = field(default_factory=set)
    acks: set = field(default_factory=set)


def _beta_entropy(a: float, b: float) -> float:
    """Differential entropy of Beta(a, b) (Eq. 4's H[Beta]); 0 for Beta(1,1)."""
    from scipy.stats import beta as _beta
    return float(_beta(a, b).entropy())


def _sim(u, v) -> float:
    """Cosine similarity of two MiniLM vectors, or word-set Jaccard when the embedder is
    unavailable (the offline smoke test), where `_embed` returns word sets."""
    if isinstance(u, frozenset) or isinstance(v, frozenset):
        if not isinstance(u, frozenset) or not isinstance(v, frozenset):
            return 0.0
        return len(u & v) / len(u | v) if (u | v) else 0.0
    return float(MACLA.st_util.cos_sim(u, v)[0][0])


def _json_from(text: str, opener: str):
    """The first complete JSON array ('[') or object ('{') in `text`, or None.

    Decodes from the opener and ignores whatever follows the value: qwen3 sometimes adds
    a stray closing bracket after a valid list (M3 run 60495244), which made the old
    first-opener-to-last-closer slice fail to parse.
    """
    text = text or ""
    dec = json.JSONDecoder()
    i = text.find(opener)
    while i >= 0:
        try:
            return dec.raw_decode(text, i)[0]
        except json.JSONDecodeError:
            i = text.find(opener, i + 1)
    return None


class MACLAMemory(MemoryModule):
    """Procedural memory after MACLA; see the module docstring for the section map."""

    name = "macla"

    def __init__(
        self,
        run_dir: Path,
        llm=None,
        N_p: int = 200,
        N_m: int = 50,
        theta_dup: float = THETA_DUP,
        theta_risk: float = 0.90,
        segment_with_llm: bool = True,
        refine_with_llm: bool = True,
        ontology_every: int = 100,
        show_locations: bool = True,
        top_k_locations: int = 5,
        abort_after: int = 3,
        decay_episodes: float = 100.0,
    ):
        self.llm = llm
        self.theta_dup = theta_dup
        self.theta_risk = theta_risk
        self.segment_with_llm = segment_with_llm
        self.refine_with_llm = refine_with_llm
        self.ontology_every = ontology_every
        self.show_locations = show_locations
        self.top_k_locations = top_k_locations
        self.abort_after = abort_after
        self.decay_episodes = decay_episodes

        # The engine supplies the stores and the ontology; its reasoner is not used (all
        # model calls go through the harness client, H11), so it is built without one.
        self.agent = LLMMACLAAgent(N_a=1000, N_s=100, N_p=N_p, N_m=N_m, use_llm=False)
        self._pin_embedder()

        # ---- learned tables (persisted) ----
        self._ack_templates: dict = defaultdict(Counter)   # verb -> Counter(template)
        self._found_at: dict = defaultdict(Counter)        # object noun -> Counter(recep)
        self._object_vocab: Counter = Counter()
        self._receptacle_vocab: Counter = Counter()
        self._proc_verb: dict = {}          # proc key -> key verb
        self._pre_held: dict = {}           # proc key -> Counter({True: n, False: m})
        self._last_used: dict = {}          # proc/meta key -> episode index
        self._refined_at: dict = {}         # proc key -> evidence (alpha+beta) at refinement
        self._meta_family: dict = {}        # meta key -> task family
        self._meta_desc: dict = {}          # meta key -> descriptor text
        # P26: the engine's MetaProcedure has no context sets; §4.7 appends every outcome's
        # initial context to the success or failure set of the procedure OR meta-procedure.
        self._meta_ctx: dict = {}           # meta key -> {"success": [...], "failure": [...]}
        self._history: dict = defaultdict(lambda: deque(maxlen=META_WINDOW))
        self._zero_shot = [0, 0]            # [successes, episodes] with no memory text
        self._ontology_corpus: list = []
        self._ontology_built_at = 0
        self._n_episodes = 0
        self._ack_cache: dict | None = None
        self._emb: dict = {}                # key -> embedding (procedures and metas)

        # ---- per-episode state ----
        self._reset_episode()

        # ---- logging only ----
        self._n_memory_calls = 0
        self._n_memory_errors = 0
        self._memory_prompt_tokens = 0
        self._memory_completion_tokens = 0
        self._counts = Counter()            # segmentation / refinement / selection events
        self.last_update_report: dict = {}

        self._ckpt_dir = Path(run_dir) / "macla"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # Set-up helpers
    # ======================================================================

    @staticmethod
    def _pin_embedder() -> None:
        """P11: load MiniLM at a fixed revision in place of the engine's unpinned copy."""
        if not MACLA._EMBED_AVAILABLE:
            return
        if getattr(MACLA, "_PINNED_REVISION", None) == MINILM_REVISION:
            return
        try:
            from sentence_transformers import SentenceTransformer
            MACLA._EMBEDDER = SentenceTransformer(MINILM, revision=MINILM_REVISION)
            MACLA._PINNED_REVISION = MINILM_REVISION
        except Exception as exc:  # pragma: no cover
            log.warning("could not load pinned MiniLM (%s); using the engine's copy", exc)

    def _reset_episode(self) -> None:
        self._ep_task = ""
        self._ep_obs: list = []
        self._ep_bind = _Bind()
        self._ep_meta: str | None = None        # meta-procedure running this episode
        self._ep_meta_aborted = False
        self._ep_meta_aborted_key: str | None = None   # P24: the meta that was aborted
        self._ep_cur = None                     # (index, key) of the running sub-procedure
        self._ep_cur_steps = 0                  # steps it has run without its result
        self._ep_meta_fails = 0                 # sub-procedure failures since last progress
        self._ep_shown: dict = {}               # procedure key -> step it was first shown
        self._ep_done: dict = {}                # meta key -> sub-procedure indices done
        self._ep_any_memory = False
        self._h_cache: tuple = ("", None)

    def _llm_call(self, system: str, user: str, max_tokens: int) -> str:
        """One metered memory-side call through the harness client (H11, DEC-08)."""
        if self.llm is None:
            return ""
        before = self.llm.token_usage
        try:
            return self.llm.complete(system, user, max_tokens=max_tokens) or ""
        except Exception as exc:  # noqa: BLE001
            self._n_memory_errors += 1
            log.error("MACLA memory call failed: %s", exc)
            return ""
        finally:
            after = self.llm.token_usage
            self._n_memory_calls += after["n_calls"] - before["n_calls"]
            self._memory_prompt_tokens += after["prompt_tokens"] - before["prompt_tokens"]
            self._memory_completion_tokens += (after["completion_tokens"]
                                               - before["completion_tokens"])

    def _embed(self, text: str):
        if not text:
            return None
        if not MACLA._EMBED_AVAILABLE:
            return frozenset(re.findall(r"[a-z<>]+", text.lower()))
        try:
            return MACLA._EMBEDDER.encode(text, convert_to_tensor=True)
        except Exception as exc:  # pragma: no cover
            log.debug("embedding failed (%s)", exc)
            return None

    # ======================================================================
    # Acknowledgments (postcondition checks) and the episode state
    # ======================================================================

    @staticmethod
    def _parse_action(action: str) -> tuple:
        low = (action or "").strip().lower()
        verb = low.split()[0] if low else ""
        return low, verb, _ARG_RE.findall(low)

    def _learn_ack(self, verb: str, args: list, obs_next: str) -> None:
        """Learn 'you <effect> the {a1} ... the {a2}' from an acknowledged action."""
        first = obs_next.split(".")[0].strip()
        if not first.startswith("you ") or not args:
            return
        tpl = first
        for i, (noun, num) in enumerate(args[:2], 1):
            tpl = tpl.replace(f"{noun} {num}", "{a%d}" % i)
        if "{a1}" not in tpl:
            return
        self._ack_templates[verb][tpl] += 1
        self._ack_cache = None

    def _ack_regexes(self) -> dict:
        """verb -> compiled regex; learned template when available, else fallback."""
        if self._ack_cache is not None:
            return self._ack_cache
        out: dict = {}
        verbs = set(_FALLBACK_ACK) | set(self._ack_templates) | set(_SIGNIFICANT)
        for verb in verbs:
            v = "move" if verb == "put" else verb
            pat = None
            if self._ack_templates.get(verb):
                tpl, _ = self._ack_templates[verb].most_common(1)[0]
                pat = re.escape(tpl)
                pat = pat.replace(re.escape("{a1}"), r"(?P<a1>[a-z]+) (?P<n1>\d+)")
                pat = pat.replace(re.escape("{a2}"), r"(?P<a2>[a-z]+) (?P<n2>\d+)")
            if pat is None:
                pat = _FALLBACK_ACK.get(v) or (
                    rf"you {re.escape(v)}[a-z]* the (?P<a1>[a-z]+) (?P<n1>\d+)"
                    rf"(?: [a-z/]+ the (?P<a2>[a-z]+) (?P<n2>\d+))?")
            if v not in out or self._ack_templates.get(verb):
                out[v] = re.compile(pat)
        self._ack_cache = out
        return out

    def _state_from(self, observations: list) -> _State:
        st = _State()
        rxs = self._ack_regexes()
        for o in observations:
            o = (o or "").lower()
            for v, rx in rxs.items():
                m = rx.search(o)
                if not m:
                    continue
                a1, n1 = m.group("a1"), m.group("n1")
                a2 = m.groupdict().get("a2")
                st.acks.add(v)
                if v == "take":
                    st.held = (a1, n1)
                    st.taken.add((a1, n1))
                    st.placed.pop((a1, n1), None)
                elif v == "move":
                    if st.held == (a1, n1):
                        st.held = None
                    if a2:
                        st.placed[(a1, n1)] = a2
                elif v in ("heat", "cool", "clean", "slice"):
                    st.treated[v].add((a1, n1))
                elif v == "use":
                    st.used.add(a1)
        return st

    # ======================================================================
    # π: cleaning and placeholders (ours; §4.1 "abstracted action sequence", §C.6)
    # ======================================================================

    def _effective(self, result: EpisodeResult) -> list:
        """Actions whose effect the environment registered, in order, with their index.

        An action counts when the admissible-command set changed, or when the next
        observation acknowledged it by naming its arguments. The final action of a
        successful episode is kept: its reply is `final_observation` (K8).
        """
        traj = result.trajectory
        n = len(traj)
        out: list = []
        for i, t in enumerate(traj):
            low, verb, args = self._parse_action(t.get("action") or "")
            if not low:
                continue
            if i + 1 < n:
                nxt = traj[i + 1]
                obs_next = (nxt.get("observation") or "").strip().lower()
                adm_now = t.get("admissible_commands") or []
                adm_next = nxt.get("admissible_commands") or []
            else:
                # The last action's reply is final_observation (K8), for failures too.
                obs_next = (getattr(result, "final_observation", "") or "").strip().lower()
                adm_now = adm_next = []
            obs_now = (t.get("observation") or "").strip().lower()
            acked = bool(args) and obs_next.startswith("you ") and all(
                f"{a} {d}" in obs_next for a, d in args)
            if adm_now and adm_next:
                changed = set(adm_now) != set(adm_next)
            else:
                changed = obs_next != obs_now and not obs_next.startswith("nothing happens")
            if changed or acked or (i + 1 == n and result.success):
                out.append(_Step(low, verb, args, acked=acked, last=(i + 1 == n), idx=i))
        return out

    @staticmethod
    def _inverse(a: _Step, b: _Step, any_copy_of: str | None = None) -> bool:
        """take X from R / move X to R (either order), open R / close R (either order).

        With `any_copy_of`, two copies of that object count as the same X (P28)."""
        if {a.verb, b.verb} in ({"take", "move"}, {"take", "put"}):
            if not (len(a.args) > 1 and len(b.args) > 1 and a.args[1] == b.args[1]):
                return False
            return a.args[0] == b.args[0] or (
                any_copy_of is not None and a.args[0][0] == b.args[0][0] == any_copy_of)
        if {a.verb, b.verb} == {"open", "close"}:
            return bool(a.args) and a.args == b.args
        return False

    def _cancel(self, steps: list, any_copy_of: str | None = None) -> list:
        """Cancel inverse pairs to a fixpoint; drop consecutive identical actions."""
        changed = True
        while changed:
            changed = False
            out: list = []
            i = 0
            while i < len(steps):
                if i + 1 < len(steps) and self._inverse(steps[i], steps[i + 1], any_copy_of):
                    i += 2
                    changed = True
                    continue
                if (i + 2 < len(steps) and self._inverse(steps[i], steps[i + 2], any_copy_of)
                        and steps[i + 1].verb == "go" and steps[i + 1].args
                        and len(steps[i].args) > 1
                        and steps[i + 1].args[0] == steps[i].args[1]):
                    i += 3
                    changed = True
                    continue
                if out and out[-1].action == steps[i].action:
                    i += 1
                    changed = True
                    continue
                out.append(steps[i])
                i += 1
            steps = out
        return steps

    @staticmethod
    def _drop_exploration(steps: list) -> list:
        """Drop navigation groups that led to no manipulation; collapse go-runs."""
        groups: list = []
        for s in steps:
            if s.verb == "go" or not groups:
                groups.append([s])
            else:
                groups[-1].append(s)
        kept = [s for g in groups
                if any(s.verb in _SIGNIFICANT for s in g) or any(s.last for s in g)
                for s in g]
        out: list = []
        for s in kept:
            if out and out[-1].verb == "go" and s.verb == "go":
                out[-1] = s
            else:
                out.append(s)
        return out

    @staticmethod
    def _first_effects_only(steps: list) -> list:
        """Keep only the first application of a persistent effect to an instance."""
        seen: set = set()
        out: list = []
        for s in steps:
            if s.verb in ("heat", "cool", "clean", "slice", "use") and s.args:
                key = (s.verb, s.args[0])
                if key in seen:
                    continue
                seen.add(key)
            out.append(s)
        return out

    def _clean(self, steps: list, obj: str | None) -> list:
        """π of a segment: first effects, inverse pairs cancelled, searches dropped, and
        only the task object's manipulations (tools and navigation stay)."""
        steps = self._drop_exploration(self._cancel(self._first_effects_only(steps)))
        if obj:
            steps = [s for s in steps
                     if s.verb not in _SIGNIFICANT or s.verb == "use" or not s.args
                     or s.args[0][0] == obj]
            # P28: π names the object only as <object>, so putting one copy down and taking
            # another from the same place is a no-op in π. Matching numbered copies alone let
            # instance swaps through ("move tomato 3 to countertop 1, take tomato 1 from
            # countertop 1"), stored as a put-down/pick-up loop (proc_000115, f1500-macla).
            steps = self._drop_exploration(self._cancel(steps, any_copy_of=obj))
        return steps

    @staticmethod
    def _room_receptacles(admissible: list, observation: str) -> set:
        recs = {m.group(1) for c in (admissible or [])
                for m in [re.match(r"go to ([a-z]+) \d+", c.lower())] if m}
        if not recs and observation:
            recs = {n for n, _ in _ARG_RE.findall(observation.lower())}
        return recs

    def _bind(self, task: str, receptacles: set) -> _Bind:
        """Resolve the task's object and goal receptacle from learned vocabularies."""
        words = re.findall(r"[a-z]+", task.lower())
        target = None
        if receptacles:
            target = next((w for w in reversed(words) if w in receptacles), None)
        if target is None:
            target = next((w for w in reversed(words) if w in self._receptacle_vocab), None)
        # P22: the object comes from the task sentence. It came from the nouns the agent had
        # ever picked up, so a task whose object type was never taken bound nothing, the
        # object filter in `_clean` was skipped, and a failed episode stored procedures about
        # whatever the agent did pick up (22/17/18 per final store, 13-22% of all evidence,
        # e.g. "take box" from "put two tissuebox in shelf"). The vocabulary is the fallback.
        parsed = extract_target_objects(task)
        obj = parsed[0] if parsed and parsed[0] != target else None
        if obj is None:
            obj = next((w for w in words if w != target and w in self._object_vocab), None)
        return _Bind(obj=obj, target=target, receptacles=set(receptacles))

    @staticmethod
    def _abstract(steps: list, obj: str | None, target: str | None) -> list:
        """Placeholder form of cleaned steps (§C.6).

        The task object becomes <object> and the task's goal receptacle <target>; the
        receptacle the object was taken from is <source>, in the pickup and in the "go
        to" that led there. A tool (the instrument of heat/cool/clean/slice/use) keeps
        its name even when it is also the target, since which tool a sub-task needs is
        part of what the procedure records.
        """
        out: list = []
        for i, s in enumerate(steps):
            tool = None
            if s.verb in ("heat", "cool", "clean", "slice") and len(s.args) > 1:
                tool = s.args[1][0]
            elif s.verb == "use" and s.args:
                tool = s.args[0][0]
            src = s.args[1][0] if s.verb == "take" and len(s.args) > 1 else None
            if s.verb == "go":
                nxt = next((t for t in steps[i + 1:] if t.verb != "go"), None)
                if nxt is not None and nxt.verb == "take" and len(nxt.args) > 1:
                    src = nxt.args[1][0]
                elif nxt is not None and nxt.verb in ("heat", "cool", "clean", "slice")                         and len(nxt.args) > 1:
                    tool = nxt.args[1][0]
                elif nxt is not None and nxt.verb == "use" and nxt.args:
                    tool = nxt.args[0][0]

            def _slot(m, tool=tool, src=src):
                noun = m.group(1)
                if noun == tool:
                    return noun
                if obj and noun == obj:
                    return "<object>"
                if noun == src:
                    return "<source>"
                if target and noun == target:
                    return "<target>"
                return noun

            out.append(_ARG_RE.sub(_slot, s.action))
        return out

    @staticmethod
    def _placeholders(text: str, bind: _Bind) -> str:
        """Instance numbers dropped and the task's object and target as placeholders."""
        t = re.sub(r"\b([a-z]+) \d+\b", r"\1", (text or "").strip().lower())
        if bind.obj:
            t = re.sub(rf"\b{re.escape(bind.obj)}s?\b", "<object>", t)
        if bind.target:
            t = re.sub(rf"\b{re.escape(bind.target)}\b", "<target>", t)
        return t

    def _abstract_text(self, text: str, bind: _Bind) -> str:
        """G, Ψ and Φ in placeholder form (§C.6), one short phrase each."""
        return self._placeholders(text, bind)[:160]

    @staticmethod
    def _key_verb(steps: list) -> str:
        for s in reversed(steps):
            v = s.split()[0] if s else ""
            if v in _SIGNIFICANT:
                return "move" if v == "put" else v
        return ""

    @staticmethod
    def _action_group(steps: list) -> str:
        """The significant actions of π in order, repeats collapsed: "take+cool+move".

        Duplicate detection compares only procedures of the same group (ours, P15). It
        was the last significant action alone, so a "cool then place" segment counted as
        a placement, merged into a plain placement and lost its cool step (Stage 6: 49
        segmentation replies mentioned cooling, 2 cool procedures survived, 0 of 63 cool
        test tasks solved).
        """
        out: list = []
        for st in steps:
            v = st.split()[0] if st else ""
            v = "move" if v == "put" else v
            if v in _SIGNIFICANT and (not out or out[-1] != v):
                out.append(v)
        return "+".join(out)

    # ======================================================================
    # §4.1 segmentation (DEC-11)
    # ======================================================================

    def _segment_llm(self, result: EpisodeResult) -> list | None:
        """[(start, end, goal, pre, post)] from one LLM call, or None if unusable."""
        traj = result.trajectory
        lines = []
        for i, t in enumerate(traj):
            reply = (traj[i + 1].get("observation") if i + 1 < len(traj)
                     else getattr(result, "final_observation", "")) or ""
            reply = " ".join(reply.split())[:160]
            lines.append(f"step {i}: > {t.get('action', '')}\n   game: {reply}")
        obs0 = " ".join((traj[0].get("observation") or "").split())[:500] if traj else ""
        outcome = "completed the task" if result.success else "did not complete the task"
        user = SEGMENT_USER.format(outcome=outcome, task=result.task_description, obs0=obs0,
                                   steps="\n".join(lines))
        reply = self._llm_call(SEGMENT_SYSTEM, user, max_tokens=900)
        items = _json_from(reply, "[")
        if not (isinstance(items, list) and any(isinstance(x, dict) for x in items)):
            # A one-sub-task episode can come back as a single object, or the list can
            # come wrapped in an object ({"segments": [...]}).
            obj = _json_from(reply, "{")
            items = None
            if isinstance(obj, dict):
                items = next((v for v in obj.values()
                              if isinstance(v, list) and v and isinstance(v[0], dict)),
                             [obj] if "start" in obj else None)
        if not isinstance(items, list):
            self._log_call("segment", result.task_id, reply, False)
            return None
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                s, e = int(it.get("start")), int(it.get("end"))
            except (TypeError, ValueError):
                continue
            if e < s:
                s, e = e, s
            out.append((max(s, 0), min(e, len(traj) - 1), str(it.get("goal") or ""),
                        [str(p) for p in (it.get("preconditions") or [])][:6],
                        [str(p) for p in (it.get("postconditions") or [])][:6]))
        self._log_call("segment", result.task_id, reply, bool(out))
        return out or None

    def _log_call(self, kind: str, ref: str, reply: str, usable: bool) -> None:
        """Keep every segmentation and refinement reply (logging only, A0.4), so a
        fallback or an unparsed refinement can be traced to what the model said."""
        try:
            with (self._ckpt_dir / "llm_calls.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"episode": self._n_episodes, "kind": kind, "ref": ref,
                                    "usable": usable, "reply": reply}) + "\n")
        except OSError as exc:  # pragma: no cover
            log.warning("could not log a MACLA memory call (%s)", exc)

    @staticmethod
    def _segment_rule(steps: list) -> list:
        """Fallback when the LLM reply is unusable: cut at navigation boundaries."""
        groups: list = []
        for s in steps:
            if s.verb == "go" or not groups:
                groups.append([s])
            else:
                groups[-1].append(s)
        segs, carry = [], []
        for g in groups:
            if not any(s.verb in _SIGNIFICANT for s in g):
                carry.extend(g)
                continue
            g = carry + g
            carry = []
            key = next(s for s in reversed(g) if s.verb in _SIGNIFICANT)
            segs.append((g[0].idx, g[-1].idx, f"{key.verb} the object", [], []))
        return segs

    # ======================================================================
    # Storage: §4.1 merge, §4.6 pruning
    # ======================================================================

    @staticmethod
    def _signature(proc: Procedure) -> str:
        """[G; Ψ; Φ] (§4.1), the text embedded for retrieval and duplicate detection."""
        return " ; ".join([proc.goal, " ".join(proc.preconditions[:6]),
                           " ".join(proc.postconditions[:6])]).strip()

    def _find_duplicate(self, proc: Procedure, verb: str, emb) -> str | None:
        """P1, P15: the most similar stored procedure with the same action group, if its
        [G; Ψ; Φ] similarity exceeds θ_dup, or one with the identical step sequence."""
        mem = self.agent.memory_system.procedural_memory
        group = self._action_group(proc.steps)
        best, best_sim = None, 0.0
        for key, entry in mem.items():
            if self._action_group(entry.procedure.steps) != group:
                continue
            if entry.procedure.steps == proc.steps:
                return key
            other = self._emb.get(key)
            if emb is None or other is None:
                continue
            sim = _sim(emb, other)
            if sim > best_sim:
                best, best_sim = key, sim
        self._counts["dup_checks"] += 1
        return best if best_sim > self.theta_dup else None

    def _add_or_merge(self, proc: Procedure, verb: str) -> tuple[str, bool]:
        emb = self._embed(self._signature(proc))
        dup = self._find_duplicate(proc, verb, emb)
        mem = self.agent.memory_system
        if dup is not None:
            p = mem.procedural_memory[dup].procedure
            for x in proc.preconditions:
                if x not in p.preconditions:
                    p.preconditions.append(x)
            for x in proc.postconditions:
                if x not in p.postconditions:
                    p.postconditions.append(x)
            p.preconditions, p.postconditions = p.preconditions[:8], p.postconditions[:8]
            self._emb[dup] = self._embed(self._signature(p))
            return dup, True
        if len(mem.procedural_memory) >= mem.N_p:
            self._prune()
        # Keys number procedures in insertion order; str hash() is salted per process,
        # which would make tie-breaks differ between replicates.
        key = f"proc_{mem.stats['procedures_added'] + 1:06d}"
        mem.procedural_memory[key] = MACLA.ProceduralMemoryEntry(procedure=proc)
        mem.stats["procedures_added"] += 1
        self._proc_verb[key] = verb
        self._emb[key] = emb
        return key, False

    def _utility(self, key: str, uses_total: int) -> float:
        """Eq. 8: 0.5 * reliability + 0.3 * frequency + 0.2 * recency (episode index)."""
        p = self.agent.memory_system.procedural_memory[key].procedure
        rel = p.alpha / (p.alpha + p.beta)
        freq = p.execution_count / max(uses_total, 1)
        rec = math.exp(-(self._n_episodes - self._last_used.get(key, 0)) / self.decay_episodes)
        return 0.5 * rel + 0.3 * freq + 0.2 * rec

    def _prune(self) -> None:
        """§4.6: evict the lowest-utility procedure from the largest key-verb cluster,
        never one a meta-procedure uses."""
        mem = self.agent.memory_system
        used = {pk for mp in mem.meta_procedural_memory.values() for pk in mp.sub_procedures}
        clusters: dict = defaultdict(list)
        for k in mem.procedural_memory:
            if k not in used:
                clusters[self._proc_verb.get(k, "")].append(k)
        if not clusters:
            return
        total = sum(e.procedure.execution_count for e in mem.procedural_memory.values())
        biggest = max(clusters.values(), key=len)
        victim = min(biggest, key=lambda k: self._utility(k, total))
        del mem.procedural_memory[victim]
        for d in (self._emb, self._proc_verb, self._pre_held, self._last_used, self._refined_at):
            d.pop(victim, None)
        self._counts["evicted"] += 1

    # ======================================================================
    # §4.2 selection
    # ======================================================================

    def _risk(self, entry, observation: str) -> float:
        """Fraction of past failures whose first observation resembles this one (Jaccard
        above θ_risk; the engine's one-sided overlap saturated at 1.0 on ALFWorld)."""
        return self._risk_of(entry.failure_contexts, observation)

    def _risk_of(self, failure_contexts: list, observation: str) -> float:
        if not failure_contexts:
            return 0.0
        a = set(observation.lower().split())
        if not a:
            return 0.0
        hits = 0
        for f in failure_contexts:
            b = set(f.observation_init.lower().split())
            u = len(a | b)
            if u and len(a & b) / u > self.theta_risk:
                hits += 1
        return hits / len(failure_contexts)

    def _eu(self, rel: float, alpha: float, beta: float, risk: float) -> float:
        """Eq. 4: Rel·ρ·R_max − Risk·(1−ρ)·C_fail + λ_info·H[Beta(α, β)]."""
        rho = alpha / (alpha + beta)
        return rel * rho * R_MAX - risk * (1 - rho) * C_FAIL + LAMBDA_INFO * _beta_entropy(alpha, beta)

    @property
    def theta_conf(self) -> float:
        """§D.1.2 Eq. 13: θ_conf = ρ_LLM·R_max − (1 − ρ_LLM)·C_fail − c_LLM (P16).

        ρ_LLM is the zero-shot success rate: the share of this arm's training episodes
        that ran with no memory text and succeeded, under a Beta(1,1) prior. Measured on
        training data only; frozen with the store for evaluation.
        """
        s, n = self._zero_shot
        rho = (s + 1) / (n + 2)
        return rho * R_MAX - (1 - rho) * C_FAIL - C_LLM

    def _h(self, task: str, observation: str):
        """h_t: the task instruction and the current observation (§4.7: the agent embeds
        its observation "and, optionally, an instruction string"; App. C retrieves
        "goal-relevant procedures"), written with the same placeholders as the stored
        procedures so a new object can match them.

        History: the observation alone (Alg. 1, lines 1 and 26) was used from commit
        5aac20b. In Stage 6 replicate 1 (job 60500946) that made retrieval ignore the task
        after step 0: on cool tasks MACLA showed desklamp and examine procedures and never
        its cool procedures (0 of 21 cool tasks). User decision 2026-09-28: include the task.
        """
        b = self._ep_bind
        key = self._placeholders(task, b) + "\n" + self._placeholders(observation, b)
        if self._h_cache[0] != key:
            self._h_cache = (key, self._embed(key))
        return self._h_cache[1]

    def _candidates(self, h) -> list:
        """Top-k procedures and meta-procedures by similarity to h (Alg. 1, line 2)."""
        mem = self.agent.memory_system
        scored = []
        for k in list(mem.procedural_memory) + list(mem.meta_procedural_memory):
            e = self._emb.get(k)
            if e is None or h is None:
                continue
            scored.append((_sim(h, e), k))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[:TOP_K]

    def _select(self, task: str, observation: str) -> tuple:
        """(key, EU, table) of the best candidate, key None when below θ_conf."""
        mem = self.agent.memory_system
        table = []
        for rel, k in self._candidates(self._h(task, observation)):
            if k in mem.meta_procedural_memory:
                if self._ep_meta_aborted:
                    continue
                mp = mem.meta_procedural_memory[k]
                # P26: Eq. 4 scores every candidate the same way, Risk included; it was 0.
                fails = self._meta_ctx.get(k, {}).get("failure", [])
                eu = self._eu(rel, mp.alpha, mp.beta, self._risk_of(fails, observation))
            else:
                e = mem.procedural_memory[k]
                eu = self._eu(rel, e.procedure.alpha, e.procedure.beta, self._risk(e, observation))
            table.append((round(eu, 3), round(rel, 3), k))
        table.sort(key=lambda x: (-x[0], x[2]))
        if not table or table[0][0] < self.theta_conf:
            return None, (table[0][0] if table else None), table
        return table[0][2], table[0][0], table

    # ======================================================================
    # §4.4 control policy and CheckPre / CheckPost
    # ======================================================================

    def _post_done(self, key: str, st: _State, bind: _Bind, n_before: int = 0) -> bool:
        """CheckPost: the key action's acknowledgment for the task object was observed.

        The n-th retrieval stage needs n objects taken and the n-th placement stage n
        objects placed at the destination (pick-two tasks do both twice).
        """
        verb = self._proc_verb.get(key, "")
        steps = self.agent.memory_system.procedural_memory[key].procedure.steps

        def mine(inst) -> bool:
            return bind.obj is None or inst[0] == bind.obj

        if verb == "take":
            return sum(1 for i in st.taken if mine(i)) > n_before
        if verb == "move":
            dest = next((s.split()[-1] for s in reversed(steps)
                         if s.split()[:1] in (["move"], ["put"])), "<target>")
            noun = bind.target if dest == "<target>" else dest
            n = sum(1 for i, loc in st.placed.items() if mine(i) and (noun is None or loc == noun))
            return n > n_before
        if verb in ("heat", "cool", "clean", "slice"):
            return any(mine(i) for i in st.treated.get(verb, ()))
        if verb == "use":
            return bool(st.used)
        return verb in st.acks

    def _pre_ok(self, key: str, st: _State, bind: _Bind) -> bool:
        """CheckPre: the learned "object held at the start" precondition matches now."""
        if not self._pre_held.get(key):
            return True
        return self._holding(st, bind) == self._needs_held(key)

    def _policy(self, meta_key: str, st: _State, bind: _Bind) -> list:
        """[(proc_key, decision)] for a meta-procedure's sub-procedures (§4.4).

        skip: its postcondition was observed this episode; a stage once done stays
              done for the rest of the episode (P2);
        continue: the first sub-procedure that is not done;
        repeat: the last finished retrieval, when the current sub-procedure needs the
                object held and it is not (it was put down too early);
        pending: the sub-procedures after the current one.
        Abort is decided in retrieve(), by `_meta_failed`, when sub-procedure failures recur.
        """
        mp = self.agent.memory_system.meta_procedural_memory[meta_key]
        mem = self.agent.memory_system.procedural_memory
        done = self._ep_done.setdefault(meta_key, set())
        out: list = []
        current = None
        seen_verb: Counter = Counter()      # n-th take / n-th placement (pick-two tasks)
        for i, pk in enumerate(mp.sub_procedures):
            if pk not in mem:
                continue
            verb = self._proc_verb.get(pk, "")
            if current is None and i not in done                     and self._post_done(pk, st, bind, n_before=seen_verb[verb]):
                done.add(i)
            seen_verb[verb] += 1
            if current is None and i in done:
                out.append((pk, "skip"))
            elif current is None:
                current = pk
                out.append((pk, "continue"))
            else:
                out.append((pk, "pending"))
        if current is not None and self._needs_held(current) and not self._holding(st, bind):
            takes = [j for j, (pk, d) in enumerate(out)
                     if d == "skip" and self._proc_verb.get(pk) == "take"]
            if takes:
                out[takes[-1]] = (out[takes[-1]][0], "repeat")
        return out

    def _meta_failed(self, meta_key: str, st: _State, bind: _Bind) -> bool:
        """Advance the running sub-procedure's step count; True when failures recur."""
        policy = self._policy(meta_key, st, bind)
        cur = next(((i, pk) for i, (pk, d) in enumerate(policy) if d == "continue"), None)
        if cur != self._ep_cur:
            if cur is None or self._ep_cur is None or cur[0] > self._ep_cur[0]:
                self._ep_meta_fails = 0          # progress: a sub-procedure finished
            self._ep_cur, self._ep_cur_steps = cur, 0
            return False
        if cur is None:
            return False
        self._ep_cur_steps += 1
        budget = max(1, len(self.agent.memory_system.procedural_memory[cur[1]].procedure.steps))
        if self._ep_cur_steps >= budget:
            self._ep_meta_fails += 1
            self._ep_cur_steps = 0
            self._counts["meta_subprocedure_failed"] += 1
        return self._ep_meta_fails >= self.abort_after

    def _needs_held(self, key: str) -> bool:
        c = self._pre_held.get(key)
        return bool(c) and c.most_common(1)[0][0]

    @staticmethod
    def _holding(st: _State, bind: _Bind) -> bool:
        return st.held is not None and (bind.obj is None or st.held[0] == bind.obj)

    # ======================================================================
    # Rendering
    # ======================================================================

    def _render_steps(self, steps: list, bind: _Bind) -> str:
        obj = bind.obj or "object"
        tgt = bind.target or "goal location"

        def _sub(m):
            slot = m.group(0)
            return {"<object>": obj, "<target>": tgt,
                    "<source>": f"where the {obj} is"}.get(slot, slot.strip("<>"))

        return " -> ".join(re.sub(r"<[a-z_]+>", _sub, s) for s in steps[:8])

    def _fill(self, text: str, bind: _Bind) -> str:
        return (text.replace("<object>", bind.obj or "object")
                .replace("<target>", bind.target or "goal location"))

    def _render_proc(self, key: str, bind: _Bind, label: str) -> str:
        p = self.agent.memory_system.procedural_memory[key].procedure
        lines = [f"{label}{self._fill(p.goal, bind)}: {self._render_steps(p.steps, bind)}"]
        if p.preconditions:
            lines.append("     when: " + "; ".join(self._fill(x, bind) for x in p.preconditions[:3]))
        if p.postconditions:
            lines.append("     done when: " + "; ".join(self._fill(x, bind) for x in p.postconditions[:3]))
        return "\n".join(lines)

    def _locations(self, bind: _Bind) -> str:
        if not (self.show_locations and bind.obj):
            return ""
        counts = Counter(self._found_at.get(bind.obj, {}))
        if not counts:
            for cat, members in (self.agent.bayesian_selector.ontology or {}).items():
                if bind.obj == cat or bind.obj in members:
                    for sib in [cat, *members]:
                        if sib != bind.obj:
                            counts.update(self._found_at.get(sib, {}))
        ranked = [r for r, _ in counts.most_common()]
        if bind.receptacles:
            ranked = [r for r in ranked if r in bind.receptacles]
        ranked = ranked[: self.top_k_locations]
        return (f"Likely locations of the {bind.obj} (where it was found before): "
                + ", ".join(ranked)) if ranked else ""

    # ======================================================================
    # MemoryModule interface
    # ======================================================================

    def retrieve(self, task: str, observation: str, admissible: list, step: int) -> MemoryContext:
        """Algorithm 1, one step: continue a running meta-procedure under its policy, or
        embed the observation, score the candidates (Eq. 4) and use the best one above
        θ_conf; below it the step is zero-shot and gets no memory text."""
        if step == 0 or task != self._ep_task:
            self._reset_episode()
            self._ep_task = task
            self._ep_bind = self._bind(task, self._room_receptacles(admissible, observation))
        low = (observation or "").lower()
        if low and (not self._ep_obs or self._ep_obs[-1] != low):
            self._ep_obs.append(low)
        bind = self._ep_bind
        if not bind.receptacles and admissible:
            bind.receptacles = self._room_receptacles(admissible, observation)
        st = self._state_from(self._ep_obs)
        mem = self.agent.memory_system
        meta: dict = {"theta_conf": round(self.theta_conf, 3)}

        # Abort (§4.4: "aborting when failures recur"). P25, user decision 2026-09-29: a
        # sub-procedure fails when its expected result (CheckPost) has not appeared after as
        # many steps as its action sketch has; `abort_after` such failures without progress
        # abort the meta-procedure. The old rule counted "Nothing happens." replies, which
        # never occur because the agent picks from the admissible actions: 0 aborts in all
        # final runs.
        if self._ep_meta and self._meta_failed(self._ep_meta, st, bind):
            meta["policy_event"] = "abort"
            self._ep_meta_aborted = True
            self._ep_meta_aborted_key = self._ep_meta
            self._ep_meta = None

        if self._ep_meta is None:
            key, eu, table = self._select(task, observation)
            meta["candidates"] = table
            meta["selected_eu"] = eu
            if key is not None and key in mem.meta_procedural_memory:
                self._ep_meta = key
                self._last_used[key] = self._n_episodes
            elif key is not None:
                if not self._pre_ok(key, st, bind):
                    meta["path"] = "precondition_failed"
                    return MemoryContext(system_context="", metadata=meta)
                self._ep_shown.setdefault(key, step)
                self._last_used[key] = self._n_episodes
                self._ep_any_memory = True
                p = mem.procedural_memory[key].procedure
                text = "\n".join(x for x in [
                    f"[MACLA PROCEDURE, reliability {p.success_rate:.2f} over "
                    f"{p.alpha + p.beta - 2} uses]",
                    self._render_proc(key, bind, "  "),
                    self._locations(bind)] if x)
                meta.update(path="procedure", key=key)
                return MemoryContext(system_context=text, metadata=meta)
            else:
                meta["path"] = "zero_shot"
                return MemoryContext(system_context="", metadata=meta)

        mk = self._ep_meta
        mp = mem.meta_procedural_memory[mk]
        policy = self._policy(mk, st, bind)
        self._ep_any_memory = True
        lines = [f"[MACLA META-PROCEDURE for this kind of task, reliability "
                 f"{mp.success_rate:.2f}]"]
        for i, (pk, d) in enumerate(policy, 1):
            if d in ("continue", "repeat"):
                self._ep_shown.setdefault(pk, step)
            tag = {"skip": "[skip: done]", "continue": "[continue]", "repeat": "[repeat]",
                   "pending": "[next]"}[d]
            if d in ("continue", "repeat"):
                lines.append(self._render_proc(pk, bind, f"  {i}. {tag} "))
            else:
                goal = self._fill(mem.procedural_memory[pk].procedure.goal, bind)
                lines.append(f"  {i}. {tag} {goal}")
        loc = self._locations(bind)
        if loc:
            lines.append(loc)
        meta.update(path="meta_procedure", key=mk, policy=[d for _k, d in policy])
        return MemoryContext(system_context="\n".join(lines), metadata=meta)

    # ---------------------------------------------------------------- learning

    def _learn_tables(self, result: EpisodeResult) -> None:
        """Acknowledgment templates, vocabularies and found-at counts, from every episode."""
        traj = result.trajectory
        last_at = None
        for i, t in enumerate(traj):
            low, verb, args = self._parse_action(t.get("action") or "")
            if not low:
                continue
            if verb == "go" and args:
                self._receptacle_vocab[args[0][0]] += 1
            if verb == "take" and args:
                self._object_vocab[args[0][0]] += 1
            obs_next = ((traj[i + 1].get("observation") if i + 1 < len(traj)
                         else getattr(result, "final_observation", "")) or "").strip().lower()
            if not (args and obs_next.startswith("you ")
                    and all(f"{a} {d}" in obs_next for a, d in args)):
                continue
            self._learn_ack(verb, args, obs_next)
            if verb == "go":
                last_at = args[0][0]
            elif verb == "take" and len(args) > 1:
                self._found_at[args[0][0]][args[1][0]] += 1
            elif verb == "use" and last_at:
                self._found_at[args[0][0]][last_at] += 1

    def _extract(self, result: EpisodeResult, bind: _Bind) -> list:
        """§4.1 for one training episode: [proc key] of its segments, in order.

        MACLA extracts from failed episodes too (§1: "failed trajectories often contain
        correct substeps"; §B.1: "effective knowledge extraction even from failed
        episodes"). From a failed episode only the correct substeps are kept: a segment
        whose key action the game acknowledged, and a placement only onto the task's
        destination (a failure that put the object down elsewhere is not a skill; P1).
        """
        if not result.success and bind.obj is None:
            # P22: without the task's object, a failure's correct substeps cannot be told
            # from its mistakes, so it teaches nothing.
            self._counts["failure_unbound"] += 1
            return []
        steps = self._effective(result)
        segs = self._segment_llm(result) if (self.segment_with_llm and self.llm) else None
        if segs is None:
            self._counts["segment_fallback"] += 1
            segs = self._segment_rule(self._clean(steps, bind.obj))
            log.info("MACLA segmentation fell back to navigation cuts for %s", result.task_id)
        else:
            self._counts["segment_llm"] += 1
        obs = [(t.get("observation") or "") for t in result.trajectory]
        # π comes from the whole trajectory cleaned once, then cut by segment, so an
        # undone pickup that the model split across two segments cannot survive.
        cleaned = self._clean(steps, bind.obj)
        keys: list = []
        for start, end, goal, pre, post in segs:
            part = [s for s in cleaned if start <= s.idx <= end]
            pi = self._abstract(part, bind.obj, bind.target)
            verb = self._key_verb(pi)
            if not verb:
                continue
            if not result.success:
                key_step = next(s for s in reversed(part) if s.verb in _SIGNIFICANT)
                if not key_step.acked:
                    self._counts["failure_segment_unverified"] += 1
                    continue
                if verb == "move" and not (bind.target and len(key_step.args) > 1
                                           and key_step.args[1][0] == bind.target):
                    self._counts["failure_segment_off_target"] += 1
                    continue
                self._counts["failure_segment_kept"] += 1
            held = self._state_from(obs[: part[0].idx + 1]).held if part else None
            held = held is not None and (bind.obj is None or held[0] == bind.obj)
            proc = Procedure(
                goal=self._abstract_text(goal, bind) or f"{verb} <object>",
                preconditions=[self._abstract_text(x, bind) for x in pre if x],
                steps=pi,
                postconditions=[self._abstract_text(x, bind) for x in post if x],
                concepts={classify_task_family(result.task_description)},
                alpha=1, beta=1, source_trajectory=result.task_id,
            )
            key, merged = self._add_or_merge(proc, verb)
            self._counts["merged" if merged else "inserted"] += 1
            self._pre_held.setdefault(key, Counter())[held] += 1
            if not keys or keys[-1] != key or verb == "move":
                keys.append(key)
        return keys

    def _maybe_meta(self, family: str, seq: list, task: str, bind: _Bind) -> str:
        """§4.4: a sequence in >= θ_meta of the family's last 15 episodes, at least two
        procedures long, becomes a meta-procedure (App. C: 3 of 15)."""
        hist = self._history[family]
        need = math.ceil(THETA_META * META_WINDOW)
        if len(seq) < 2 or sum(1 for h in hist if h == tuple(seq)) < need:
            return "none"
        mem = self.agent.memory_system
        for k, fam in self._meta_family.items():
            if fam == family and k in mem.meta_procedural_memory \
                    and list(mem.meta_procedural_memory[k].sub_procedures) == list(seq):
                return "exists"
        n = sum(1 for h in hist if h == tuple(seq))
        procs = mem.procedural_memory
        mp = MetaProcedure(
            goal_meta=self._abstract_text(task, bind),
            preconditions_meta=list(procs[seq[0]].procedure.preconditions[:3]),
            sub_procedures=list(seq),
            composition_policy={"type": "control_policy",
                                "branches": ["continue", "skip", "repeat", "abort"]},
            alpha=1 + n, beta=1,       # Beta(1,1) plus the successes that formed it
        )
        mk = mem.add_meta_procedure(mp)
        self._meta_family[mk] = family
        desc = " ; ".join([mp.goal_meta] + [procs[k].procedure.goal for k in seq])
        self._meta_desc[mk] = desc
        self._emb[mk] = self._embed(desc)
        self._meta_family = {k: v for k, v in self._meta_family.items()
                             if k in mem.meta_procedural_memory}
        return "created"

    def update(self, result: EpisodeResult) -> None:
        if self._read_only:
            return
        before = Counter(self._counts)
        calls_before = self._n_memory_calls
        self._n_episodes += 1
        family = classify_task_family(result.task_description)
        mem = self.agent.memory_system
        self._learn_tables(result)
        self._ontology_corpus.append({"task": result.task_description,
                                      "actions": [t["action"] for t in result.trajectory][:20]})
        self._ontology_corpus = self._ontology_corpus[-600:]
        if (self._n_episodes - self._ontology_built_at >= self.ontology_every
                or not self._ontology_built_at):
            try:
                self.agent.bayesian_selector.build_ontology(self._ontology_corpus, k_top=100)
                self._ontology_built_at = self._n_episodes
            except Exception as exc:  # pragma: no cover
                log.debug("ontology build failed (%s)", exc)

        if not self._ep_any_memory:
            self._zero_shot[1] += 1
            self._zero_shot[0] += int(result.success)

        obs = [t.get("observation", "") for t in result.trajectory]
        if getattr(result, "final_observation", ""):
            obs.append(result.final_observation)
        st = self._state_from(obs)
        bind = self._ep_bind if self._ep_task == result.task_description else _Bind()
        if bind.obj is None:
            adm0 = result.trajectory[0].get("admissible_commands") if result.trajectory else []
            bind = self._bind(result.task_description,
                              self._room_receptacles(adm0, obs[0] if obs else ""))
        ctx = ContrastiveContext(
            observation_init=obs[0] if obs else "",
            action_sequence=[t["action"] for t in result.trajectory],
            observation_term=obs[-1] if obs else "",
            cumulative_reward=1.0 if result.success else 0.0,
            trajectory_id=result.task_id, success=result.success)

        # ---- Bayesian credit (Alg. 1, lines 12 and 20), once per procedure per episode --
        # P27: CheckPost judges the state the procedure's execution produced, so only the
        # replies after the step it was first shown count. obs[s] is the observation the
        # agent saw at step s, before acting; its own action's reply is obs[s + 1]. Judging
        # the whole episode credited an object taken before the procedure appeared: two
        # take-procedures that are only selected once the object is held reached 54-0 and
        # 57-0 while 6/53 and 23/56 of their episodes succeeded (f1500-macla).
        credit: dict = {}
        for k, s in self._ep_shown.items():
            if k in mem.procedural_memory:
                credit[k] = self._post_done(k, self._state_from(obs[s + 1:]), bind)
        # Alg. 1, lines 11-12: the meta-procedure's outcome updates (alpha, beta) whether it
        # ran to completion or was aborted (P24: an aborted one got no update at all); an
        # abort is a failure. Its initial context joins its success or failure set (P26).
        ran = self._ep_meta or self._ep_meta_aborted_key
        if ran and ran in mem.meta_procedural_memory:
            mp = mem.meta_procedural_memory[ran]
            ok = bool(result.success) and self._ep_meta_aborted_key != ran
            mp.alpha += int(ok)
            mp.beta += int(not ok)
            mp.execution_count += 1
            sets = self._meta_ctx.setdefault(ran, {"success": [], "failure": []})
            side = "success" if ok else "failure"
            sets[side] = (sets[side] + [ctx])[-FAIL_CAP:]

        # ---- §4.1 extraction from every episode (verified substeps of failures) ------
        seq: list = self._extract(result, bind) if result.trajectory else []
        for k in seq:
            credit[k] = True           # its effect was observed: one success (P19: not two)
        # Meta-procedures compose whole-task playbooks, so only successful episodes
        # count toward the 3-of-15 recurrence; a failure still takes a window slot.
        meta_seq = seq if result.success else []
        self._history[family].append(tuple(meta_seq))
        meta_event = (self._maybe_meta(family, meta_seq, result.task_description, bind)
                      if meta_seq else "none")

        for k, ok in credit.items():
            if k not in mem.procedural_memory:
                continue
            e = mem.procedural_memory[k]
            if ok:
                e.procedure.alpha += 1
                e.success_contexts = (e.success_contexts + [ctx])[-FAIL_CAP:]
            else:
                e.procedure.beta += 1
                e.failure_contexts = (e.failure_contexts + [ctx])[-FAIL_CAP:]
            e.procedure.execution_count += 1
            self._last_used[k] = self._n_episodes

        # ---- §4.3 refinement --------------------------------------------------------
        for k, e in list(mem.procedural_memory.items()):
            if len(e.success_contexts) < N_MIN or len(e.failure_contexts) < N_MIN:
                continue
            # P12: re-refine after 3 more uses, counted on alpha+beta, which keeps growing
            # after the capped context lists are full.
            evidence = e.procedure.alpha + e.procedure.beta
            if evidence - self._refined_at.get(k, -N_MIN) < N_MIN:
                continue
            self._refine(k, e)
            self._refined_at[k] = evidence

        self._reset_episode()
        self.last_update_report = {
            **{k: v - before.get(k, 0) for k, v in self._counts.items() if v - before.get(k, 0)},
            "segments_stored": len(seq),
            "meta_procedure": meta_event,
            "credited": len(credit),
            "credited_success": sum(1 for v in credit.values() if v),
            "memory_llm_calls": self._n_memory_calls - calls_before,
            "theta_conf": round(self.theta_conf, 3),
            "n_procedures_after": len(mem.procedural_memory),
            "n_meta_procedures_after": len(mem.meta_procedural_memory),
        }

    def _refine(self, key: str, entry) -> None:
        """§4.3, Eq. 6: Ψ ← Ψ ∪ Ψ+ \\ Ψ−; π ← Merge(π, Δπ); Φ ← Φ ∪ ΔΦ.

        π is replaced only by a list whose every verb appears in π or in the successful
        uses' own actions, so the model cannot introduce an action nobody took.
        """
        p = entry.procedure
        if not (self.refine_with_llm and self.llm):
            return
        def _uses(ctxs):
            return "\n".join(
                f"- {' '.join(c.observation_init.split())[:300]} | actions: "
                f"{'; '.join(c.action_sequence[:25])}" for c in ctxs[-4:])
        reply = self._llm_call(REFINE_SYSTEM, REFINE_USER.format(
            goal=p.goal, pre="; ".join(p.preconditions), steps=" -> ".join(p.steps),
            post="; ".join(p.postconditions), succ=_uses(entry.success_contexts),
            fail=_uses(entry.failure_contexts)), max_tokens=500)
        out = _json_from(reply, "{")
        self._counts["refine_calls"] += 1
        self._log_call("refine", key, reply, isinstance(out, dict))
        if not isinstance(out, dict):
            self._counts["refine_unparsed"] += 1
            return
        pos = [str(x).lower()[:160] for x in (out.get("preconditions_positive") or [])][:3]
        neg = {str(x).lower()[:160] for x in (out.get("preconditions_negative") or [])}
        p.preconditions = [x for x in p.preconditions + pos if x not in neg]
        p.preconditions = list(dict.fromkeys(p.preconditions))[:8]
        for x in (out.get("postconditions") or [])[:3]:
            x = str(x).lower()[:160]
            if x not in p.postconditions:
                p.postconditions.append(x)
        p.postconditions = p.postconditions[:8]
        # P23: π is shown joined with " -> " and qwen3 sometimes returns it that way, as one
        # string; that string was accepted as a single step and never repaired.
        new_steps = [part.strip().lower() for s in (out.get("steps") or [])
                     for part in str(s).split("->") if part.strip()]
        allowed = {s.split()[0] for s in p.steps if s} | {
            a.split()[0].lower() for c in entry.success_contexts for a in c.action_sequence if a}
        if new_steps and all(s.split()[0] in allowed for s in new_steps) \
                and self._action_group(new_steps) == self._action_group(p.steps):
            if new_steps[:12] != p.steps:
                self._counts["refine_steps_changed"] += 1
            p.steps = new_steps[:12]
        entry.discriminative_patterns = {"preconditions_positive": pos,
                                         "preconditions_negative": sorted(neg)}
        self._emb[key] = self._embed(self._signature(p))
        self.agent.memory_system.stats["procedures_refined"] += 1

    # ---------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        mem = self.agent.memory_system
        for store in (mem.atomic_memory, mem.procedural_memory, mem.meta_procedural_memory,
                      mem.context_index, mem.goal_index):
            store.clear()
        for k in mem.stats:
            mem.stats[k] = 0
        mem._meta_counter = 0
        self.__init__(run_dir=self._ckpt_dir.parent, llm=self.llm,
                      N_p=mem.N_p, N_m=mem.N_m, theta_dup=self.theta_dup,
                      theta_risk=self.theta_risk, segment_with_llm=self.segment_with_llm,
                      refine_with_llm=self.refine_with_llm,
                      ontology_every=self.ontology_every, show_locations=self.show_locations,
                      top_k_locations=self.top_k_locations, abort_after=self.abort_after,
                      decay_episodes=self.decay_episodes)

    def finalize(self) -> None:
        if self._read_only:
            return
        self.save(self._ckpt_dir / "state.pkl")

    @property
    def n_procedures(self) -> int:
        return len(self.agent.memory_system.procedural_memory)

    def stats(self) -> dict:
        eps = max(self._n_episodes, 1)
        return {
            **self.get_stats(),
            "memory_llm_calls": self._n_memory_calls,
            "memory_prompt_tokens": self._memory_prompt_tokens,
            "memory_completion_tokens": self._memory_completion_tokens,
            "memory_llm_calls_per_episode": round(self._n_memory_calls / eps, 3),
            "memory_llm_errors": self._n_memory_errors,
            "events": dict(self._counts),
            # P11: None means the pinned MiniLM did not load and the engine's copy is used.
            "embedder_revision": getattr(MACLA, "_PINNED_REVISION", None),
        }

    def get_stats(self) -> dict:
        mem = self.agent.memory_system
        procs = list(mem.procedural_memory.values())
        return {
            "n_procedures": len(procs),
            "n_meta_procedures": len(mem.meta_procedural_memory),
            "n_episodes": self._n_episodes,
            "avg_posterior": round(sum(e.procedure.success_rate for e in procs) / len(procs), 3)
            if procs else 0.0,
            "n_refined": mem.stats.get("procedures_refined", 0),
            "n_ontology_categories": len(self.agent.bayesian_selector.ontology),
            "theta_conf": round(self.theta_conf, 3),
            "zero_shot_episodes": self._zero_shot[1],
            "procedures_by_verb": dict(Counter(self._proc_verb.get(k, "") for k in
                                               mem.procedural_memory)),
        }

    def save(self, path: Path) -> None:
        mem = self.agent.memory_system
        state = {
            "procedural_memory": dict(mem.procedural_memory),
            "meta_procedural_memory": dict(mem.meta_procedural_memory),
            "meta_counter": mem._meta_counter,
            "proc_verb": dict(self._proc_verb),
            "pre_held": {k: dict(v) for k, v in self._pre_held.items()},
            "last_used": dict(self._last_used),
            "refined_at": dict(self._refined_at),
            "meta_family": dict(self._meta_family),
            "meta_desc": dict(self._meta_desc),
            "meta_ctx": {k: {s: list(v) for s, v in d.items()} for k, d in self._meta_ctx.items()},
            "history": {f: list(h) for f, h in self._history.items()},
            "zero_shot": list(self._zero_shot),
            "n_episodes": self._n_episodes,
            "ontology_corpus": list(self._ontology_corpus),
            "ack_templates": {v: dict(c) for v, c in self._ack_templates.items()},
            "found_at": {n: dict(c) for n, c in self._found_at.items()},
            "object_vocab": dict(self._object_vocab),
            "receptacle_vocab": dict(self._receptacle_vocab),
        }
        with Path(path).open("wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        mem = self.agent.memory_system
        mem.procedural_memory.update(state.get("procedural_memory", {}))
        mem.meta_procedural_memory.update(state.get("meta_procedural_memory", {}))
        mem._meta_counter = max(mem._meta_counter, int(state.get("meta_counter", 0) or 0))
        self._proc_verb.update(state.get("proc_verb", {}))
        self._pre_held.update({k: Counter(v) for k, v in (state.get("pre_held") or {}).items()})
        self._last_used.update(state.get("last_used", {}))
        self._refined_at.update(state.get("refined_at", {}))
        self._meta_family.update(state.get("meta_family", {}))
        self._meta_desc.update(state.get("meta_desc", {}))
        self._meta_ctx.update(state.get("meta_ctx") or {})
        for f, h in (state.get("history") or {}).items():
            self._history[f].extend(tuple(x) for x in h)
        self._zero_shot = list(state.get("zero_shot", [0, 0]))
        self._n_episodes = state.get("n_episodes", 0)
        self._ontology_corpus = list(state.get("ontology_corpus", []))
        for v, c in (state.get("ack_templates") or {}).items():
            self._ack_templates[v].update(c)
        for n, c in (state.get("found_at") or {}).items():
            self._found_at[n].update(c)
        self._object_vocab.update(state.get("object_vocab") or {})
        self._receptacle_vocab.update(state.get("receptacle_vocab") or {})
        self._ack_cache = None
        # Embeddings are recomputed, not pickled: they depend only on the stored text.
        for k, e in mem.procedural_memory.items():
            self._emb[k] = self._embed(self._signature(e.procedure))
        for k, d in self._meta_desc.items():
            self._emb[k] = self._embed(d)
        if self._ontology_corpus:
            try:
                self.agent.bayesian_selector.build_ontology(self._ontology_corpus, k_top=100)
                self._ontology_built_at = self._n_episodes
            except Exception as exc:  # pragma: no cover
                log.debug("ontology build failed (%s)", exc)
