"""Knowledge-guided planner, after Optimus-1 (DEC-10).

Optimus-1 plans once per task: g1..gn = p(o, t, p(t)) (Eq. 2, p.5-6), where o is the
observation, t the task and p(t) the sub-graph retrieved for the task. Its prompt
(Listing 1, App. E.4) gives the task, the observation and the retrieved "craft graph",
and asks for a plan in the format of a worked example.

ALFWorld version, one memory-side LLM call per episode, at its first step:
  * the observation is the game's text, so Optimus-1's first turn, which reads the game
    screen into text ("visual inference"), is not needed and is merged away;
  * the retrieved sub-graph is the plan block (`HDKG.build_context`): the learned
    sub-goal order, tool and locations for the task object;
  * the worked example uses placeholders only, so the prompt carries no task facts.

The planner's numbered sub-goals replace the plan block in the memory text, as Optimus-1's
controller receives sub-goals rather than the graph. If the call fails or returns no
numbered line, the caller serves the plan block itself.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

SYSTEM = ("You are an expert in a text-based household game and you can guide an agent "
          "to complete tasks.")

USER = """Now you need to make a plan with the help of <observation> and <knowledge>.
<observation>: what the agent sees now. Based on it, consider whether preliminary steps
       are needed so that the agent can complete the task.
<knowledge>: what the agent's memory learned from earlier tasks of this kind: the usual
       order of sub-goals, the tool used, and where the object is usually found.
I will give you an example of the output format as follows:
[Example]
<task>: <the task>
<task planning>:
1. <first sub-goal>
2. <second sub-goal>
3. <further sub-goals, one per line, in order>
Here are the task, observation and knowledge. You MUST output in example format: the
numbered sub-goals under <task planning>, and nothing else.
<task>: {task}
<observation>: {observation}
<knowledge>:
{knowledge}
<task planning>:"""

_STEP = re.compile(r"^\s*(\d+)[.)]\s+(.*\S)\s*$")
MAX_SUBGOALS = 12


def parse_plan(reply: str) -> list[str]:
    """The numbered sub-goal lines of a reply, in order, without their numbers."""
    goals = []
    for line in (reply or "").splitlines():
        m = _STEP.match(line)
        if m:
            goals.append(m.group(2))
    return goals[:MAX_SUBGOALS]


class KnowledgePlanner:
    def __init__(self, llm, max_tokens: int = 400) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.stats = {"calls": 0, "failures": 0, "unparsed": 0}
        self.last_reply = ""

    def plan(self, task: str, observation: str, knowledge: str) -> list[str]:
        """Sub-goals for this episode, or [] when the call fails or yields none."""
        self.stats["calls"] += 1
        user = USER.format(task=task, observation=" ".join((observation or "").split()),
                           knowledge=knowledge or "(nothing learned yet)")
        try:
            self.last_reply = self.llm.complete(SYSTEM, user, max_tokens=self.max_tokens) or ""
        except Exception:
            log.exception("planner call failed")
            self.stats["failures"] += 1
            self.last_reply = ""
            return []
        goals = parse_plan(self.last_reply)
        if not goals:
            self.stats["unparsed"] += 1
        return goals
