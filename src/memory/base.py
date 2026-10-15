"""Memory module interface and shared dataclasses."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MemoryContext:
    """What the memory module provides to the agent before action selection."""

    system_context: str = ""
    # Arm-specific diagnostics (which pages were read, which plan was chosen, ...). Never
    # shown to the agent; recorded at step 0 in EpisodeResult.memory_metadata.
    metadata: dict = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """What the agent reports to the memory module after an episode."""

    task_id: str
    task_family: str
    task_description: str
    trajectory: list[dict]  # [{step, action, observation, admissible_commands, score}]
    success: bool
    steps_taken: int
    scene_id: str | None = None
    # Diagnostic fields — populated by agent.py from the memory module.
    memory_context: str = ""         # memory text injected at step 0 (after the budget cap)
    memory_metadata: dict = field(default_factory=dict)  # MemoryContext.metadata at step 0
    # Per-episode memory-injection telemetry, summed over steps. Measured in agent.py at
    # the single point every arm passes through, so the numbers are comparable across
    # arms by construction rather than by each arm agreeing to report the same thing.
    # Counted with the Qwen3 tokenizer (src/tokens.py); runner.compute_summary has the
    # aggregate. Keys: steps, retrieval_steps, offered_tokens, injected_tokens,
    # discarded_tokens, truncated_steps, peak_offered_tokens.
    memory_injection: dict = field(default_factory=dict)
    # Logging only (H12). Never read by an arm, and kept out of `trajectory` on purpose:
    # some arms serialise the trajectory into their own prompts, so adding keys there
    # would change what those arms see.
    game_file: str = ""              # the ALFWorld game this episode played
    # The observation after the last action (K8). The trajectory holds each step's
    # observation BEFORE its action, so without this the result of the final action,
    # usually the one that completed the task, is lost. A field of its own, not a
    # trajectory entry, so arms that serialise the trajectory see nothing new.
    final_observation: str = ""
    # The memory text actually served, recorded at step 0 and again whenever it changes:
    # [{step, text, metadata, offered_tokens, injected_tokens}].
    memory_log: list[dict] = field(default_factory=list)
    # One entry per step: [{step, offered_tokens, injected_tokens, prompt_tokens,
    # completion_tokens, action_calls, pick_path, raw_output}]. prompt_tokens and
    # completion_tokens are the server's own counts for the action call(s) of that step.
    step_log: list[dict] = field(default_factory=list)
    system_prompt: str = ""          # the action model's system prompt this episode
    # Wall time in seconds spent in retrieve(), the action model and update().
    memory_timing: dict = field(default_factory=dict)
    # What update() wrote to the store this episode, as the arm reports it
    # (last_update_report). Empty at eval, where update() does nothing.
    memory_update: dict = field(default_factory=dict)


class MemoryModule(ABC):
    """Interface that all memory backends implement."""

    # Set by freeze(); read by update()/finalize() to suppress all writes during eval.
    # Defined on the base so every arm inherits a working freeze without re-implementing
    # a private flag (the per-arm `read_only` config key it replaces was unenforceable
    # at the harness level — runner.py never set it on most arms).
    _read_only: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for logging/configs."""
        ...

    def freeze(self) -> None:
        """Put the memory into read-only mode for the evaluation phase.

        After this is called, update() and finalize() must not mutate the store.
        run_pipeline.py calls this once for EVERY arm at the train->eval boundary, so
        the eval-freeze is enforced uniformly rather than per-arm. Arms that keep extra
        write paths (e.g. RagMemory's index, KgMemory's learned priors) must check
        self._read_only before writing.
        """
        self._read_only = True

    @property
    def read_only(self) -> bool:
        return self._read_only

    @abstractmethod
    def retrieve(
        self,
        task_description: str,
        observation: str,
        admissible_commands: list[str],
        step: int,
    ) -> MemoryContext:
        """Called before each action. Returns context to inject into the LLM prompt."""
        ...

    @abstractmethod
    def update(self, episode: EpisodeResult) -> None:
        """Called after each episode. Updates the memory store."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated memory (for cold-start experiments)."""
        ...

    def finalize(self) -> None:
        """Called once after the training phase completes. Default: no-op.

        Override to run any end-of-training processing (e.g. a final batch
        synthesis pass that uses all accumulated diagnoses regardless of the
        usual minimum-count threshold).
        """
        pass
