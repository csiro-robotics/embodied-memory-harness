"""End-to-end wiring check for all four arms, with no ALFWorld and no Ollama.

For each arm it runs the real `run_pipeline.main()` (train phase, freeze, eval phase)
from the arm's real config, with three things replaced:

  * ALFWorldEnv -> the scripted FakeEnv from smoke_wiki_v3.py (a tiny solvable room),
  * build_llm   -> a scripted FakeLLM that answers the action prompt, the wiki prompts
                   and the KG triplet-extraction prompt,
  * ollama / the embedding server -> fakes, so MACLA's module import and the KG's
                   embedding calls run their real code paths offline. MACLA uses the
                   real MiniLM from the local Hugging Face cache. MACLA's model calls
                   go through the scripted FakeLLM, as on the cluster (H11).

Then it runs scripts/check_run.py on each output folder, plus the extra checks below.
The success rates mean nothing: the environment is trivial and the model is scripted.
This checks that every arm builds a store, loads it at eval, stays frozen at eval, and
that the run folder records what each arm served.

    python scripts/smoke_all_arms.py            # all arms
    python scripts/smoke_all_arms.py macla      # one arm
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

ARMS = ("no_memory", "kg_hybrid", "macla", "wiki_v3")
N_EPISODES = 12          # per phase; enough for one wiki consolidation pass at K=10


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _install_fake_ollama() -> None:
    """A stand-in `ollama` module, so MACLA.py imports without the real client."""
    if "ollama" in sys.modules:
        return
    mod = types.ModuleType("ollama")

    class _Msg(dict):
        def __init__(self, content: str):
            super().__init__(content=content)
            self.content = content

    class _Resp(dict):
        def __init__(self, content: str):
            msg = _Msg(content)
            super().__init__(message=msg)
            self.message = msg
            self.prompt_eval_count = 10
            self.eval_count = 1

    def chat(model=None, messages=None, **kw):
        return _Resp("")

    mod.chat = chat
    sys.modules["ollama"] = mod


class FakeEmbedder:
    """Hashed bag-of-words vectors: deterministic, and similar texts get similar vectors."""

    DIM = 64

    def __init__(self, *a, **k):
        pass

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for w in re.findall(r"[a-z]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def _make_fake_llm_class():
    from smoke_wiki_v3 import FakeLLM as WikiFakeLLM

    class FakeLLM(WikiFakeLLM):
        """Adds the KG extraction prompt and the fields runner.compute_summary reads."""

        def complete(self, system, user, max_tokens=None, **kw):
            if "knowledge-graph" in system.lower() or "triplet" in system.lower():
                self.n_calls += 1
                self.memory_calls += 1
                return self._triplets(user)
            return super().complete(system, user, max_tokens=max_tokens, **kw)

        @staticmethod
        def _triplets(user):
            out = []
            m = re.search(r"On the (\w+ \d+), you see a (\w+) 1", user)
            if m:
                out.append({"subject": f"{m.group(2)} 1", "relation": "is-at", "object": m.group(1)})
                out.append({"subject": m.group(2), "relation": "found-at",
                            "object": m.group(1).split()[0]})
            return json.dumps(out)

        @property
        def token_usage(self):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    "n_calls": self.n_calls}

    return FakeLLM


def _make_env_factory():
    from smoke_wiki_v3 import FakeEnv

    def factory(*, train_eval="train", max_steps=50, seed=42, num_episodes=10, **kw):
        # A different seed per split so eval sees different tasks from training.
        return FakeEnv(seed=seed + (0 if train_eval == "train" else 1000), n=num_episodes)

    return factory


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------

def run_arm(arm: str, out: Path) -> list[str]:
    import run_pipeline
    import memory.hybrid_kg_memory as kgm

    FakeLLM = _make_fake_llm_class()
    run_pipeline.ALFWorldEnv = _make_env_factory()
    run_pipeline.build_llm = lambda cfg, role="action": FakeLLM()
    kgm.OllamaEmbedder = FakeEmbedder

    argv = sys.argv
    sys.argv = ["run_pipeline.py", "--config", str(ROOT / "configs" / f"arm_{arm}.yaml"),
                "--out", str(out), "--smoke", str(N_EPISODES)]
    try:
        run_pipeline.main()
    finally:
        sys.argv = argv
    return _extra_checks(arm, out)


def _read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _extra_checks(arm: str, out: Path) -> list[str]:
    """Checks check_run.py does not make: the store is loaded, and the log shows it."""
    fails = []
    for phase in ("train", "eval"):
        if not (out / phase / "episodes.jsonl.gz").exists():
            fails.append(f"{phase}/episodes.jsonl.gz missing (H12)")
    if not (out / "pipeline.log").exists():
        fails.append("pipeline.log missing (H12)")
    ev = _read_jsonl(out / "eval" / "episodes.jsonl.gz")
    for e in ev:
        if len(e.get("step_log", [])) != e["steps_taken"]:
            fails.append("step_log does not have one entry per step (H12)")
            break
    if arm != "no_memory" and not any(e.get("memory_log") for e in ev):
        fails.append("eval episodes have no memory_log entries (H12)")
    if any(not s.get("user_prompt") for e in ev for s in e.get("step_log", [])):
        fails.append("a step_log entry has no user_prompt")
    if arm != "no_memory":
        tr = _read_jsonl(out / "train" / "episodes.jsonl.gz")
        if not all(e.get("memory_update") for e in tr):
            fails.append("a training episode has no memory_update report")
        if any(e.get("memory_update") for e in ev):
            fails.append("an eval episode reports a memory update: the store is not frozen")
        ts = json.loads((out / "train" / "summary.json").read_text())
        if not ts.get("memory_update_totals"):
            fails.append("train summary has no memory_update_totals")
    if arm == "wiki_v3":
        # Only when a consolidation pass ran in this short smoke run.
        cons = out / "train" / "wiki_v3" / "consolidations.jsonl"
        if cons.exists() and not (out / "train" / "wiki_v3" / "consolidation_detail.jsonl").exists():
            fails.append("wiki consolidated but wrote no consolidation_detail.jsonl (W9)")
    served = [e for e in ev if e.get("memory_context")]
    if arm == "no_memory":
        if served:
            fails.append("no_memory served memory text at eval")
    else:
        if not (out / "train" / arm).is_dir():
            fails.append(f"no trained store at train/{arm}")
        if not served:
            fails.append("eval episodes record no memory text: the trained store did not "
                         "reach the agent")
    for e in ev:
        if "memory_metadata" not in e:
            fails.append("episodes.jsonl lacks memory_metadata")
            break
    return fails


def main() -> int:
    # MACLA's retrieval and duplicate test use MiniLM similarity (Stage 5). The pinned
    # revision must already be in the Hugging Face cache; offline mode stops the
    # download checks. With the embedder off, the word-overlap fallback scores stay
    # below Eq. 13's threshold on this easy fake room, so eval would show no memory.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _install_fake_ollama()
    from check_run import check

    arms = [a for a in sys.argv[1:] if a in ARMS] or list(ARMS)
    if "macla" in arms:
        import MACLA  # noqa: F401  (loads the embedder before the arm pins its revision)
    tmp = Path(tempfile.mkdtemp(prefix="smoke_all_arms_"))
    results = {}
    try:
        for arm in arms:
            print(f"\n{'#' * 74}\n#  {arm}\n{'#' * 74}")
            out = tmp / arm
            fails = run_arm(arm, out)
            ok, notes = check(out)
            results[arm] = (ok and not fails, notes, fails)

        print(f"\n{'=' * 74}\n  SMOKE RESULT (wiring only; success rates mean nothing)\n{'=' * 74}")
        all_ok = True
        for arm, (ok, notes, fails) in results.items():
            all_ok &= ok
            print(f"\n  {arm}: {'PASS' if ok else 'FAIL'}")
            for n in notes:
                if not n.startswith("ok"):
                    print("    " + n)
            for f in fails:
                print("    FAIL  " + f)
        print(f"\n  {'ALL ARMS PASS' if all_ok else 'SOME ARMS FAILED'}")
        return 0 if all_ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
