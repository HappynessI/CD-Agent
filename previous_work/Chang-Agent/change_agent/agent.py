"""Agent interfaces and deterministic test/baseline implementations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from .state import AgentAction, AgentObservation


class Agent(Protocol):
    def act(self, observation: AgentObservation) -> tuple[str, AgentAction]: ...


class ScriptedAgent:
    def __init__(self, actions: Iterable[AgentAction]):
        self._actions = iter(actions)

    def act(self, observation: AgentObservation) -> tuple[str, AgentAction]:
        action = next(self._actions)
        # Scripted actions already use validated pixel coordinates. Model Agents
        # return their raw normalized-coordinate JSON in the first tuple item.
        return "", action
