"""Memory module package - ABC and factory for the four arms of the benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from memory.base import EpisodeResult, MemoryContext, MemoryModule  # noqa: F401
from memory.no_memory import NoMemory

MEMORY_TYPES = ("no_memory", "kg_hybrid", "macla", "wiki_v3")

# The files each arm's training writes and its evaluation reads, relative to
# trained_dir/<memory_type>/ (H13). Evaluation stops if any is missing: the KG and MACLA
# used to start from an empty store without a word, which scores like no_memory and looks
# like a result.
TRAINED_STORE_FILES = {
    "kg_hybrid": ("graph.json", "episodic.jsonl"),
    "macla": ("state.pkl",),
    "wiki_v3": ("journal.jsonl", "priors.json", "evidence.json", "pages"),
}


def check_trained_store(memory_type: str, trained_dir: Path) -> None:
    """Raise FileNotFoundError if the trained store of `memory_type` is incomplete."""
    base = Path(trained_dir) / memory_type
    missing = [f for f in TRAINED_STORE_FILES.get(memory_type, ()) if not (base / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"trained {memory_type} store in {base} is missing {', '.join(missing)}; "
            f"evaluation would start from an empty store"
        )


def create_memory(
    memory_type: str,
    *,
    run_dir: Path,
    llm: Any = None,
    config: dict | None = None,
    trained_dir: Path | None = None,
) -> MemoryModule:
    """Factory: instantiate a MemoryModule by name.

    run_dir:     where this phase persists its memory (training writes here).
    trained_dir: the TRAIN phase output dir, passed at eval so every arm loads its trained
                 store from trained_dir/<memory_type>/. None during training.
    """
    config = config or {}
    if trained_dir is not None:
        check_trained_store(memory_type, trained_dir)

    if memory_type == "no_memory":
        return NoMemory()

    if memory_type == "wiki_v3":
        from memory.wiki_v3_memory import WikiV3Memory

        # Training builds the store under run_dir/wiki_v3/; eval reads the one training
        # produced, under trained_dir/wiki_v3/.
        base = trained_dir if trained_dir is not None else run_dir
        return WikiV3Memory(wiki_dir=base / "wiki_v3", llm=llm, config=config)

    if memory_type == "kg_hybrid":
        from memory.hybrid_kg_memory import HybridKGMemory

        # Eval loads the trained graph/episodic/index from trained_dir/kg_hybrid/;
        # training cold-starts empty and persists to run_dir/kg_hybrid/. The index
        # SAVE path always stays under run_dir so eval never writes into trained_dir.
        # Explicit config paths (graph_path/episodic_path) override the convention,
        # e.g. to evaluate against a pre-built graph artifact.
        art = trained_dir / "kg_hybrid" if trained_dir is not None else None
        graph_path = config.get("graph_path") or ((art / "graph.json") if art else None)
        episodic_path = config.get("episodic_path") or ((art / "episodic.jsonl") if art else None)
        return HybridKGMemory(
            llm=llm,
            graph_path=graph_path,
            episodic_path=episodic_path,
            vectors_path=config.get("vectors_path"),  # None -> alongside graph_path
            vectors_save_path=run_dir / "kg_hybrid" / "triplet_vectors.json",
            learn_online=bool(config.get("learn_online", True)),
            episodic_search=bool(config.get("episodic_search", True)),
            planner=bool(config.get("planner", True)),
            depth=int(config.get("depth", 2)),
            width=int(config.get("width", 5)),
            k_steps=int(config.get("k_steps", 3)),
            top_k=int(config.get("top_k", 5)),
            embed_base_url=config.get("embed_base_url"),
            embed_model=config.get("embed_model", "nomic-embed-text"),
            persist_dir=run_dir / "kg_hybrid",
        )

    if memory_type == "macla":
        from memory.macla_memory import MACLAMemory

        mem = MACLAMemory(
            run_dir=run_dir,
            llm=llm,
            N_p=config.get("N_p", 200),
            N_m=config.get("N_m", 50),
            theta_dup=config.get("theta_dup", 0.85),
            theta_risk=config.get("theta_risk", 0.90),
            segment_with_llm=config.get("segment_with_llm", True),
            refine_with_llm=config.get("refine_with_llm", True),
            ontology_every=config.get("ontology_every", 100),
            show_locations=config.get("show_locations", True),
            top_k_locations=int(config.get("top_k_locations", 5)),
            abort_after=int(config.get("abort_after", 3)),
            decay_episodes=float(config.get("decay_episodes", 100)),
        )
        # Eval: reload the procedures learned during training.
        if trained_dir is not None:
            mem.load(trained_dir / "macla" / "state.pkl")
        return mem

    raise ValueError(
        f"Unknown memory type: {memory_type!r}. Valid: {', '.join(MEMORY_TYPES)}"
    )
