# A Controlled Evaluation of Memory Architectures for Embodied Long-Horizon Planning

Code, run logs and analysis scripts for the ACRA 2026 paper of the same name.

Raphael Schwalb, Duleesha Gunaratne, Lucas Cooper, Jeremia Yovinus, Leimin Tian, Dana Kulić,
Madhawa Perera, Brandon Matthews\*, Kavindie Katuwandeniya\* (Monash University, Australia;
\*equal supervision).

## What this repository does

The paper asks which kind of memory helps an LLM agent most on long household tasks. It
compares three memory architectures in ALFWorld, a text-based household benchmark, under
one framework in which the memory is the only part that changes.

| Arm | Config | What the memory is |
|---|---|---|
| `wiki_v3` (Wiki) | `configs/arm_wiki_v3.yaml` | Our structured LLM wiki: Markdown pages the model writes from its experience and reads before acting. |
| `kg_hybrid` (KG) | `configs/arm_kg_hybrid.yaml` | A knowledge graph. AriGraph builds and searches the graph, and Optimus-1 plans from it. |
| `macla` (PM) | `configs/arm_macla.yaml` | A procedural memory following MACLA. |
| `no_memory` (control) | `configs/arm_no_memory.yaml` | The same agent with no memory. |

Every arm runs the same agent, model (Qwen3 14B through Ollama), prompt and settings. In each
of three replicates, a fresh memory learns from 500 training games, is frozen, and is then
tested on the 134 `valid_unseen` games, which are set in rooms the agent never trained in.

## Main results

Evaluation success on the 134 unseen games, as the mean and range over three replicates.
The last column is a single run with 1,500 training games.

| Arm | 500 training games | 1,500 training games |
|---|---|---|
| Wiki | 50.5% [41.0, 55.2] | 53.7% |
| KG | 49.5% [41.0, 56.0] | 53.0% |
| PM | 21.9% [20.9, 22.4] | 33.6% |
| No memory | 11.4% [11.2, 11.9] | 11.2% |

Two follow-up experiments on the Wiki use copies of its trained stores, so nothing is
retrained. Removing the ranked search order from the Wiki's briefing lowers success by 16.7
percentage points (from 50.5% to 33.8%). Rewriting by hand the steps of one faulty page in
replicate 3 raises that replicate's examine tasks from 4 to 17 of 18.

## Layout

```
run_pipeline.py            training phase, freeze, evaluation phase for one arm
configs/
  _base.yaml               shared settings; a guard rejects arm configs that change them
  arm_<arm>.yaml           the memory settings of each arm
  arm_wiki_v3_nosearch.yaml  the search-order ablation (evaluation only)
src/
  agent.py                 one episode: retrieve memory, build the prompt, act, update memory
  runner.py                runs a phase, writes the logs, computes the summary
  llm.py                   OpenAI-compatible client for Ollama; counts calls and tokens
  alfworld_env.py          ALFWorld wrapper and the fixed, seeded game order
  protocol_config.py       config inheritance and the guard on shared settings
  prompts/act.py           the action prompt that every arm shares
  memory/                  the memory interface and one adapter per arm
  kg_hybrid/               knowledge-graph internals
  wiki_v3/                 Wiki internals
scripts/
  final_stats.py           recomputes every run-derived number in the paper (uses stats_core.py)
  check_run.py             checks finished run folders (store built, evaluation frozen, costs)
  smoke_all_arms.py        end-to-end wiring check for every arm, with no ALFWorld or Ollama
  smoke_wiki_v3.py         the same check for the Wiki, in more depth
  build_train500.py        builds the list of 500 training games
  stage_train_set.py       lays out a training set as ALFWorld's train split
analysis/                  the log analyses behind Sections 6 and 7, and their saved outputs
results/
  final/                   the runs reported in the paper (see results/README.md)
  ablation/                the two Wiki follow-up experiments
assets/                    the Qwen3 tokenizer and the list of training games
data/                      the list of 1,500 training games
slurm/                     scripts to run the arms on a SLURM cluster
tests/                     unit tests (pytest)
docs/                      notes on ALFWorld and on running Qwen3 with Ollama
```

## MACLA.py is not included

The PM arm runs the MACLA authors' engine, `MACLA.py`. Their repository has no licence
file, so we do not redistribute it. To run the PM arm or its tests, download it into `src/`:

```bash
curl -L -o src/MACLA.py \
  https://raw.githubusercontent.com/S-Forouzandeh/MACLA-LLM-Agents-AAMAS-2026-Conference/main/MACLA.py
```

Our adapter, `src/memory/macla_memory.py`, changes parts of the engine's behaviour at run
time, and its docstring lists each change. The other three arms, the tests of those arms and every
analysis script run without `MACLA.py`.

## Install

```bash
conda env create -f env.yml
conda activate wiki-pipeline
alfworld-download                     # ALFWorld game files; set ALFWORLD_DATA to their folder
ollama pull qwen3:14b
ollama pull nomic-embed-text          # kg_hybrid only
```

These checks need neither ALFWorld nor Ollama:

```bash
python -m pytest -q
python scripts/smoke_all_arms.py no_memory kg_hybrid wiki_v3   # add macla once MACLA.py is in src/
```

## Reproduce the paper's numbers from the logs

The logs of every reported run are in `results/`, so the numbers can be recomputed
without running a model.

```bash
python scripts/final_stats.py            # tables and statistics; rewrites results/final/paper_numbers.json
python analysis/sec6_analysis.py         # Section 6: per-family tests, objects never seen, PM procedures
python analysis/examine_analysis.py      # Section 6: the examine tasks of the three Wiki replicates
python analysis/figdata.py               # data behind the cost and learning-curve figures

export ALFWORLD_DATA=/path/to/alfworld   # the folder that holds json_2.1.1/
python analysis/sec7_analysis.py         # Section 7: did the memory name where the object starts?
python analysis/ablation_analysis.py     # the search-order ablation and the hand-corrected page
```

The last two scripts read each game's start state from the ALFWorld game files. The saved
output of every analysis script is in `analysis/outputs/`.

## Run the experiments again

Locally, with an Ollama server on `localhost:11434` (or set `OLLAMA_BASE_URL`):

```bash
python run_pipeline.py --config configs/arm_wiki_v3.yaml --out results/runs/wiki_v3
python run_pipeline.py --config configs/arm_wiki_v3.yaml --out results/runs/smoke --smoke 3
```

`--smoke N` caps both phases at N episodes and marks the output as not a result.
`--replicate N` sets the replicate, which changes the order of the games. `--skip-train`
evaluates an existing store, which is how the Wiki follow-up experiments ran.

On a SLURM cluster:

```bash
cp slurm/site.env.example slurm/site.env      # fill in your paths and account once
for ARM in no_memory kg_hybrid macla wiki_v3; do slurm/submit.sh $ARM; done
python scripts/check_run.py $RUNS/arm-*/
```

For the 1,500-game runs, `scripts/stage_train_set.py` copies the games listed in
`data/train1500_gamefiles.txt` into a folder. Then queue each arm with
`TRAIN_TASKS=<that folder> TRAIN_EPS=1500 EVAL_SEEN=140 slurm/submit.sh <arm>`.

The training set is the fixed list of 500 games in `assets/train500_games.txt`, which
`arm.slurm` mounts as ALFWorld's train split. The seed in `_base.yaml` fixes the game
order, so every arm sees the same games in the same order. The paper's runs used one
NVIDIA L40S GPU per run.

## What a run writes

```
<out>/
  pipeline_config.yaml     the merged config the run used
  report.json              training and evaluation summaries together
  train/
    <arm>/                 the trained memory store
    episodes.jsonl         one line per episode; episodes.jsonl.gz is the compressed copy
    diagnostics.jsonl      a short record of each episode, with its game file
    memory_growth.json     store size every 50 episodes
    summary.json           success rates, per-family rates, token and call costs, store size
  eval/                    the same files for the frozen evaluation phase
```

Each line of `episodes.jsonl` holds the trajectory, the memory text each step received,
every prompt, the token counts and time of every call, and what the memory's update wrote.
A cluster run also writes `run_env.txt`, which records the GPU, git commit, Ollama version,
model digests and Python packages.

Code comments cite internal design-decision tags (for example D17 or H13). They refer to
the project's working notes, which are not part of this repository.

## Credits

- MACLA: <https://github.com/S-Forouzandeh/MACLA-LLM-Agents-AAMAS-2026-Conference>.
- The knowledge-graph arm is built from AriGraph (<https://github.com/AIRI-Institute/AriGraph>)
  and Optimus-1 (<https://github.com/JiuTian-VL/Optimus-1>).
- ALFWorld: <https://github.com/alfworld/alfworld>.
- `assets/qwen3_tokenizer/` is the Qwen3 tokenizer, under its own licence in that folder.

## Citation

```bibtex
@inproceedings{schwalb2026memory,
  title     = {A Controlled Evaluation of Memory Architectures for Embodied Long-Horizon Planning},
  author    = {Schwalb, Raphael and Gunaratne, Duleesha and Cooper, Lucas and Yovinus, Jeremia and
               Tian, Leimin and Kuli{\'c}, Dana and Perera, Madhawa and Matthews, Brandon and
               Katuwandeniya, Kavindie},
  booktitle = {Proceedings of the Australasian Conference on Robotics and Automation (ACRA)},
  year      = {2026}
}
```

## Licence

MIT, see `LICENSE`. The licence covers our code, not `MACLA.py` or the Qwen3 tokenizer.
