#!/usr/bin/env python3
"""Dependency-light two-round smoke test for the Change-Agent control loop."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from change_agent.adapters.omniovcd_adapter import InitializationResult, MaskPairProcessor
from change_agent.agent import ScriptedAgent
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor
from change_agent.runner import ChangeAgentRunner
from change_agent.state import AgentAction
from change_agent.verifier import RuleBasedVerifier


class MockBackend:
    def __init__(self):
        self.processor = MaskPairProcessor()

    def initialize(self, t1_image, t2_image, query):
        mask1 = np.zeros(t1_image.shape[:2], dtype=bool)
        mask2 = np.zeros_like(mask1)
        mask1[2:6, 2:6] = True
        update = self.processor.rebuild(
            mask1, mask2, {"change_confidence": np.full(mask1.shape, 0.9)}
        )
        return InitializationResult(mask1, mask2, update)

    def rebuild(self, t1_mask, t2_mask, evidence):
        return self.processor.rebuild(t1_mask, t2_mask, evidence)


class MockPoint:
    def refine(self, image, initial_mask, coordinate, is_positive, click_history=()):
        result = initial_mask.copy()
        x, y = coordinate
        result[y, x] = is_positive
        return result


class MockBox:
    def segment_box(self, image, box_cxcywh_normalized, query):
        height, width = image.shape[:2]
        cx, cy, bw, bh = box_cxcywh_normalized
        x1, x2 = round((cx - bw / 2) * width), round((cx + bw / 2) * width)
        y1, y2 = round((cy - bh / 2) * height), round((cy + bh / 2) * height)
        result = np.zeros((height, width), dtype=bool)
        result[y1 : y2 + 1, x1 : x2 + 1] = True
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    args = parser.parse_args()
    image1 = np.zeros((16, 16, 3), dtype=np.uint8)
    image2 = np.zeros_like(image1)
    environment = ChangeAgentEnvironment(
        MockBackend(),
        ActionExecutor(MockPoint(), MockBox()),
        RuleBasedVerifier(accept_threshold=0.6, min_change_ratio=0.0),
        max_steps=3,
        run_metadata={"dataset_split": "synthetic-smoke", "seed": 42},
    )
    observation = environment.reset(image1, image2, "building change")
    agent = ScriptedAgent(
        [
            AgentAction("t2", "box", box=(2, 2, 6, 6)),
            AgentAction("t2", "finish"),
        ]
    )
    ChangeAgentRunner(environment, agent).run(observation)
    path = environment.trajectory.save(args.output)
    print(f"trajectory={path}")
    print(f"best_step={environment.trajectory.best_entry.step_index}")


if __name__ == "__main__":
    main()
