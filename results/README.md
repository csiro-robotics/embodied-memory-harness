# Run logs

Every run reported in the paper. Each folder name ends with the SLURM job number.

## final/

| Runs | What they are |
|---|---|
| `f500-r{1,2,3}-wiki_v3-*` | Wiki, 500 training games, replicates 1 to 3 |
| `f500-r{1,2,3}-kg_hybrid-*` | KG, 500 training games |
| `f500-r{1,2,3}-macla-p28-*` | PM, 500 training games |
| `r{1,2,3}-no_memory-*` | Control, 500 training games |
| `f1500-*` | One run per arm with 1,500 training games, evaluated on `valid_unseen` (`eval/`) and `valid_seen` (`eval_seen/`) |
| `paper_numbers.json`, `paper_numbers.txt` | The output of `scripts/final_stats.py` |

The control runs come from an earlier launch than the memory arms' runs. The control's
code path did not change between the two launches.

Each run folder holds:

- `pipeline_config.yaml` and `arm_config.yaml`, the settings the run used;
- `report.json`, the training and evaluation summaries;
- `run_env.txt`, the GPU, git commit, Ollama version, model digests and Python packages;
- `ollama_context_check.txt`, a check that the model server kept the full prompt;
- per phase, `summary.json`, `diagnostics.jsonl` (one short line per episode) and
  `memory_growth.json` (store size every 50 training episodes);
- `eval/episodes.jsonl.gz`, the full record of every evaluation episode: actions, replies,
  the memory text the agent received, every prompt and the cost of every call.

Trained stores, where they fit:

- Wiki: `train/wiki_v3/pages/`, the Markdown pages after training;
- KG: `train/kg_hybrid/graph.json`, the trained graph;
- PM: `train/macla/state.pkl` for replicate 1 only. Loading it needs `MACLA.py`.

The full training logs (`train/episodes.jsonl.gz`, 3.5 to 25 MB per run) and the complete
stores are not included because of their size. They are available from the authors.

## ablation/

The Wiki follow-up experiments. Each one evaluates a copy of a trained Wiki store from
`final/` with `--skip-train`, so no store was retrained.

| Run | What changed |
|---|---|
| `nosearch-r{1,2,3}` | No ranked search order in the briefing (`configs/arm_wiki_v3_nosearch.yaml`), on `valid_unseen` |
| `r3-seen` | Replicate 3's unedited store on `valid_seen`, the baseline for `fix2-r3` |
| `fix-r3` | Replicate 3's examine page with the faulty step deleted, on `valid_unseen` |
| `fix2-r3` | Replicate 3's examine page with its steps rewritten by hand, on `valid_unseen` and `valid_seen` |

`fix-r3` and `fix2-r3` keep the page before the edit (`examine-in-light.before.md`) and
after it (`train/wiki_v3/pages/examine-in-light.md`). `fix-r3` was not run on `valid_seen`.

## Paths in the logs

Game files and folders in the logs carry the paths of the cluster the runs used, such as
`/home/<user>/.../json_2.1.1/valid_unseen/<task>/<trial>/game.tw-pddl`. The analysis scripts
match games by the part after `valid_unseen/` or `valid_seen/`, so the logs work on any
machine.
