from __future__ import annotations

import re
from typing import Dict, Optional

from pydantic import BaseModel, root_validator


class AgentState(BaseModel):
    question: str
    thought: str
    action: Optional[str]
    action_input: str

    class Constant:
        eclaitic_answers = {"none", "null", "", "final_answer", "end"}
        # Define a regex pattern to match variations of "final answer"
        final_answer_pattern = re.compile(
            r"""
            .*           # Match any characters (zero or more) before "final answer"
            \s*          # Match zero or more whitespace characters
            end       # Match the word "answer"
            \s*          # Match zero or more whitespace characters
            .*           # Match any characters (zero or more) after "final answer"
        """,
            re.IGNORECASE | re.VERBOSE,
        )

    @classmethod
    def _is_done(cls, action: Optional[str]) -> bool:
        return (
            not action or action.lower() in AgentState.Constant.eclaitic_answers or AgentState.Constant.final_answer_pattern.match(action)
        )

    @root_validator(pre=True, allow_reuse=True)
    def _validation(cls, values: Dict) -> Dict:
        if "question" not in values or "thought" not in values or "action_input" not in values:
            raise ValueError(f"Validation some inputs are null values:{values}")

        question = values.get("question")
        thought = values.get("thought")
        action_input = values.get("action_input")
        action: str = values.get("action", None)

        if cls._is_done(action):
            action = None

        values["question"] = str(question)
        values["thought"] = str(thought)
        values["action"] = action
        values["action_input"] = str(action_input)

        return values
