"""Multi-round feedback loop orchestration."""

from __future__ import annotations

from .agent import Agent
from .environment import ChangeAgentEnvironment
from .state import AgentObservation, ChangeState


class ChangeAgentRunner:
    def __init__(self, environment: ChangeAgentEnvironment, agent: Agent):
        self.environment = environment
        self.agent = agent

    def run(self, initial_observation: AgentObservation) -> ChangeState:
        observation = initial_observation
        while not self.environment.done:
            raw, action = self.agent.act(observation)
            # Raw model output is parsed again at the Environment trust boundary.
            if raw:
                observation, _ = self.environment.step(raw)
            else:
                observation, _ = self.environment.step(action)
        return self.environment.best_state
