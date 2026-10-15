"""End-to-end shakedown for wiki_v3 with no ALFWorld and no Ollama.

WHAT THIS IS FOR. It drives the real code — `agent.run_episode`, `WikiV3Memory`, `Reader`,
`Consolidator`, both gates, the real `observation_priors` regex and the real
`memory.scene.shortlist` — against a fake environment and a fake model. It answers the
questions a unit test cannot:

  * does the briefing actually reach the executor's prompt?
  * does consolidation fire at exactly K and compile the right batch?
  * does the deterministic observation channel make the agent behave differently?
  * does `freeze()` really stop every write at the train->eval boundary?
  * how many memory-side LLM calls per episode does the thing actually make?

WHAT IT IS NOT. It is not a result. The fake environment is trivially solvable and the
fake model is scripted, so the success rate here means nothing about ALFWorld. Anything
printed below is a WIRING check.

    python scripts/smoke_wiki_v3.py
"""
from __future__ import annotations

import random
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import run_episode  # noqa: E402
from memory.wiki_v3_memory import WikiV3Memory  # noqa: E402

RECEPTACLES = (
    ["countertop 1", "countertop 2", "countertop 3"]
    + [f"cabinet {i}" for i in range(1, 7)]
    + ["fridge 1", "sinkbasin 1", "microwave 1", "diningtable 1"]
)

# family -> (verb, instrument, destination)
FAMILIES = {
    "pick_and_place": (None, None, "diningtable 1"),
    "pick_cool_then_place": ("cool", "fridge 1", "countertop 1"),
    "pick_heat_then_place": ("heat", "microwave 1", "countertop 1"),
    "pick_clean_then_place": ("clean", "sinkbasin 1", "countertop 1"),
}

# The regularity the memory is supposed to discover. Nothing tells the agent this.
HOME = {"apple": "countertop 3", "mug": "cabinet 2", "potato": "countertop 2"}
TASK_TEXT = {
    "pick_and_place": "put a {obj} in diningtable",
    "pick_cool_then_place": "cool some {obj} and put it in countertop",
    "pick_heat_then_place": "heat some {obj} and put it in countertop",
    "pick_clean_then_place": "clean some {obj} and put it in countertop",
}


class FakeEnv:
    """Mimics ALFWorldEnv's surface. Observations match the real extractor's regexes."""

    def __init__(self, seed: int = 0, n: int = 40):
        self.rng = random.Random(seed)
        self.n = n
        self._i = 0
        self._reset_state()

    def _reset_state(self):
        self.obj = self.rng.choice(list(HOME))
        self.family = self.rng.choice(list(FAMILIES))
        self.verb, self.instrument, self.dest = FAMILIES[self.family]
        # 80% of the time the object is at its usual home — a learnable regularity.
        self.loc = (
            HOME[self.obj] if self.rng.random() < 0.8 else self.rng.choice(RECEPTACLES)
        )
        self.at = None
        self.holding = None
        self.transformed = False
        self.done = False
        self._current_task_desc = TASK_TEXT[self.family].format(obj=self.obj)

    def reset(self):
        self._reset_state()
        self._i += 1
        obs = (
            "You are in the middle of a room. Looking quickly around you, you see "
            + ", ".join(f"a {r}" for r in RECEPTACLES)
            + f".\n\nYour task is to: {self._current_task_desc}"
        )
        return obs, self._admissible()

    def _admissible(self):
        cmds = [f"go to {r}" for r in RECEPTACLES]
        if self.at and self.holding is None and self.at == self.loc:
            cmds.append(f"take {self.obj} 1 from {self.at}")
        if self.holding:
            if self.verb and self.at == self.instrument and not self.transformed:
                cmds.append(f"{self.verb} {self.obj} 1 with {self.instrument}")
            if self.at:
                cmds.append(f"put {self.obj} 1 in/on {self.at}")
        return cmds + ["look", "inventory"]

    def step(self, action: str):
        action = (action or "").strip()
        obs, score = "Nothing happens.", 0.0

        if action.startswith("go to "):
            self.at = action[len("go to ") :].strip()
            if self.at == self.loc and self.holding is None:
                obs = f"You arrive at loc 1. On the {self.at}, you see a {self.obj} 1."
            else:
                obs = f"You arrive at loc 1. On the {self.at}, you see nothing."
        elif action.startswith("take "):
            self.holding = self.obj
            obs = f"You pick up the {self.obj} 1 from the {self.at}."
        elif self.verb and action.startswith(f"{self.verb} "):
            self.transformed = True
            obs = f"You {self.verb} the {self.obj} 1 using the {self.instrument}."
        elif action.startswith("put "):
            if self.holding and self.at == self.dest and (self.transformed or not self.verb):
                self.done, score = True, 1.0
                obs = f"You put the {self.obj} 1 in/on the {self.at}. Task success."
            else:
                obs = f"You put the {self.obj} 1 in/on the {self.at}."
                self.holding = None
        return obs, score, self.done, self._admissible()

    @property
    def task_description(self):
        return self._current_task_desc

    @property
    def task_family(self):
        return self.family

    @property
    def task_id(self):
        return f"smoke_{self._i}"


class FakeLLM:
    """Scripted stand-in for LLMClient. Dispatches on which prompt it was handed."""

    model = "fake:smoke"

    def __init__(self):
        self.n_calls = 0
        self.memory_calls = 0
        self.action_calls = 0

    # --------------------------------------------------------- memory path
    def complete(self, system, user, max_tokens=None, **kw):
        self.n_calls += 1
        self.memory_calls += 1
        if "which memory pages are worth reading" in system:
            return self._select(user)
        if "short briefing for a household robot" in system:
            return self._compose(user)
        if "maintain a small wiki" in system:
            return self._analyse(user)
        if "write one page of a wiki" in system:
            return self._rewrite(user)
        return ""

    @staticmethod
    def _select(user):
        task = re.search(r"TASK:\s*(.+)", user)
        task = task.group(1).lower() if task else ""
        verb = next((v for v in ("cool", "heat", "clean") if v in task), "place")
        for line in user.splitlines():
            m = re.match(r"-\s+([a-z0-9-]+)\s+\[", line.strip())
            if m and verb in m.group(1):
                return m.group(1)
        return "NONE"

    @staticmethod
    def _compose(user):
        # The real prompt tells the model to repeat the ranked search order verbatim.
        # Doing exactly that is what makes the deterministic channel reach the executor.
        order = re.search(r"SEARCH ORDER[^\n]*\n(.+)", user)
        task = re.search(r"TASK:\s*(.+)", user)
        parts = []
        if task:
            parts.append(f"Task: {task.group(1).strip()}.")
        if order:
            parts.append(f"Check in order: {order.group(1).strip()}.")
        parts.append("Take the exact object named, then complete the required step.")
        return " ".join(parts)

    @staticmethod
    def _analyse(user):
        fams = sorted(set(re.findall(r"· (pick_\w+) ·", user)))
        blocks = []
        for fam in fams[:3]:
            verb = next((v for v in ("cool", "heat", "clean") if v in fam), None)
            page = f"{verb}-before-placement" if verb else "find-take-place"
            blocks.append(
                f"PAGE: {page}\nTYPE: procedure\nFAMILIES: {fam}\nOBJECTS: \n"
                f"LESSON: Take the exact object, then complete the required step, then place it.\n"
                f"EVIDENCE: this batch\n"
            )
        return "\n".join(blocks)

    @staticmethod
    def _rewrite(user):
        name = re.search(r"PAGE TO WRITE:\s*(\S+)", user).group(1)
        fam = re.search(r"FAMILIES?:\s*(\S+)", user)
        fam = fam.group(1) if fam else ""
        return (
            f"---\nname: {name}\ndescription: How to handle {fam or 'these'} tasks end to end.\n"
            f"type: procedure\napplies_to: [{fam}]\nobjects: []\nsources: [batch]\n"
            f"confidence: attested\n---\n\n"
            "# Steps\n"
            "1. Find and take the exact object named in the task.\n"
            "2. Complete the required transformation step.\n"
            "3. Place it at the destination named in the task.\n\n"
            "# Not established\n- Whether the order of the last two steps can be swapped.\n"
        )

    # --------------------------------------------------------- action path
    def pick_action(self, system, user, admissible):
        """A deliberately literal executor: it uses the briefing if one reached it.

        This is the point of the smoke run. If the briefing does NOT reach the prompt, the
        agent falls back to scanning receptacles in listing order and does measurably
        worse — so the success rate here is a wiring signal, not a capability claim.
        """
        self.n_calls += 1
        self.action_calls += 1

        for a in admissible:                       # finish whenever possible
            if a.startswith(("take ", "cool ", "heat ", "clean ")):
                return a
        task = re.search(r"TASK:\s*(.+)", user)
        task_l = task.group(1).lower() if task else ""
        holding = "you are holding" in user.lower() or "you pick up" in user.lower()

        if holding:
            dest = "diningtable 1" if "diningtable" in task_l else "countertop 1"
            for a in admissible:
                if a.startswith("put ") and dest in a:
                    return a
            if f"go to {dest}" in admissible:
                return f"go to {dest}"

        # Prefer any receptacle instance the memory block named, in the order given.
        block = user.split("ADMISSIBLE", 1)[0]
        visited = set(re.findall(r"You arrive at loc 1\. On the ([\w ]+\d),", user))
        for name in re.findall(r"\b([a-z]+ \d+)\b", block):
            if name in visited:
                continue
            cmd = f"go to {name}"
            if cmd in admissible:
                return cmd
        for a in admissible:                       # fall back to listing order
            if a.startswith("go to ") and a[len("go to "):] not in visited:
                return a
        return admissible[0]

    @property
    def token_usage(self):
        return {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": self.n_calls}


def banner(msg):
    print(f"\n{'=' * 74}\n  {msg}\n{'=' * 74}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="wiki_v3_smoke_"))
    ok = True
    try:
        K, N_TRAIN, N_EVAL = 10, 30, 10
        llm = FakeLLM()
        mem = WikiV3Memory(tmp / "store", llm=llm, config={"consolidate_every": K})
        env = FakeEnv(seed=7)

        banner(f"PHASE 1 — TRAIN  ({N_TRAIN} episodes, consolidate_every={K})")
        results = []
        for i in range(N_TRAIN):
            before = mem.state.pass_count
            r = run_episode(env=env, memory=mem, llm=llm, max_steps=30,
                            history_window=20, memory_block_max_tokens=1024)
            results.append(r.success)
            tag = "  <-- CONSOLIDATED" if mem.state.pass_count > before else ""
            print(f"  ep {i+1:2d}  {'OK ' if r.success else 'FAIL'}  "
                  f"steps={r.steps_taken:2d}  pages={len(mem.pages.names())}{tag}")

        mem.finalize()
        st = mem.stats()

        banner("CHECKS")

        def check(label, cond, detail=""):
            nonlocal ok
            ok = ok and bool(cond)
            print(f"  [{'PASS' if cond else 'FAIL'}] {label}{('  — ' + detail) if detail else ''}")

        check("consolidation fired every K episodes",
              mem.state.pass_count == N_TRAIN // K,
              f"{mem.state.pass_count} passes over {N_TRAIN} episodes at K={K}")
        check("journal holds every episode",
              mem.journal.n_records == N_TRAIN, f"{mem.journal.n_records} records")
        check("observation channel is UNGATED (D08)",
              st["priors_observations"] > 0,
              f"{st['priors_observations']} observations, {st['priors_objects']} objects")
        check("action channel is SUCCESS-GATED (D08)",
              st["evidence_forms"] > 0, f"{st['evidence_forms']} attested forms")
        check("pages were written and stayed small (D52)",
              0 < st["n_pages"] <= 20, f"{st['n_pages']} pages")
        check("index regenerated and is consistent",
              "By task family" in mem.pages.read_index())
        check("read path costs ~2 LLM calls/episode (D54)",
              1.0 <= st["read_llm_calls"] / N_TRAIN <= 2.0,
              f"{st['read_llm_calls'] / N_TRAIN:.2f} read calls/ep")
        check("memory-side cost is far below v2's tens of calls",
              st["memory_llm_calls_per_episode"] < 4.0,
              f"{st['memory_llm_calls_per_episode']} calls/ep total")
        # D17 names INSTANCES of the best-ranked receptacle TYPE, in room-listing order.
        # It does not rank within a type, because `observation_priors` keys by type — see
        # the "known limitations" note in the internal design doc. Inherited from the
        # shared cross-arm mechanism, not introduced here.
        _sl = mem.priors.shortlist("apple", " ".join(f"a {r}" for r in RECEPTACLES))
        check("shortlist names receptacle INSTANCES, not types (D17)",
              bool(_sl) and all(any(c.isdigit() for c in n) for n in _sl), str(_sl))
        check("lint reports no unreachable pages",
              not any("unreachable" in p for p in mem.pages.lint()),
              "; ".join(mem.pages.lint()) or "clean")

        banner("PHASE 2 — EVAL (frozen)")
        mem.freeze()
        j_before, p_before = mem.journal.n_records, mem.state.pass_count
        calls_before = llm.memory_calls
        eval_env = FakeEnv(seed=99, n=N_EVAL)
        ev = []
        for i in range(N_EVAL):
            r = run_episode(env=eval_env, memory=mem, llm=llm, max_steps=30,
                            history_window=20, memory_block_max_tokens=1024)
            ev.append(r.success)
        mem.finalize()

        check("freeze() stops every journal write",
              mem.journal.n_records == j_before, f"{mem.journal.n_records} vs {j_before}")
        check("freeze() stops consolidation",
              mem.state.pass_count == p_before)
        check("the read path still runs while frozen",
              llm.memory_calls > calls_before,
              f"{llm.memory_calls - calls_before} memory calls during eval")

        banner("SUMMARY (wiring only — these rates mean nothing about ALFWorld)")
        print(f"  train SR   {sum(results)}/{len(results)}    eval SR   {sum(ev)}/{len(ev)}")
        print(f"  pages      {st['n_pages']}  {st['by_type']}")
        print(f"  priors     {st['priors_objects']} objects / {st['priors_observations']} observations")
        print(f"  evidence   {st['evidence_forms']} attested action forms")
        print(f"  memory LLM {st['memory_llm_calls_per_episode']} calls/episode")
        print(f"  store      {tmp / 'store'}")
        print("\n  pages written:")
        for n in mem.pages.names():
            pg = mem.pages.read(n)
            print(f"    {n}.md  [{pg.type}, {pg.word_count()}w]  applies_to={pg.applies_to}")

        banner("RESULT: ALL CHECKS PASSED" if ok else "RESULT: SOME CHECKS FAILED")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
