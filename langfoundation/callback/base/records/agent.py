from __future__ import annotations

from enum import auto
from typing import Optional

from strenum import LowercaseStrEnum

from langfoundation.callback.base.records.base import BaseRecord
from langfoundation.callback.base.records.tool import ToolRecord


class AgentState(LowercaseStrEnum):
    THOUGHT = auto()
    END = auto()


class AgentRecord(BaseRecord):
    """
    Represents an agent record.

    An agent record includes the current state of the agent, a log of the agent's actions,
    and, optionally, a tool record and a result.

    The agent may not use any tools in certain scenarios, resulting in cases where agent_start and on_tool are skipped.

    Some examples of these scenarios include:
    - When the model is basing its actions on its existing knowledge
    - When some context is missing
    - When there is already some context in history
    - When the model already possesses the necessary knowledge to take action

    Attributes:
        state (AgentState): The current state of the agent.
        log (str): A log of the agent's actions.
        tool (Optional[ToolRecord]): The tool record associated with the agent, if any.
        result (Optional[dict]): The result of the agent's actions, if any.
    """

    state: AgentState
    log: str

    tool: Optional[ToolRecord] = None
    result: Optional[dict] = None
