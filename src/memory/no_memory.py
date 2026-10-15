"""NoMemory - ReAct baseline with no persistent cross-episode memory."""
from __future__ import annotations

from memory.base import EpisodeResult, MemoryContext, MemoryModule


class NoMemory(MemoryModule):
    """Returns empty context. The agent sees only the current observation."""

    @property
    def name(self) -> str:
        return "no_memory"

    def retrieve(
        self,
        task_description: str,
        observation: str,
        admissible_commands: list[str],
        step: int,
    ) -> MemoryContext:
        return MemoryContext()

    def update(self, episode: EpisodeResult) -> None:
        pass

    def reset(self) -> None:
        pass
