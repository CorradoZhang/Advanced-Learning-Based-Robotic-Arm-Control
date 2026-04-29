from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panda_cube_grasp import PandaCubeGraspEnv


HOME_ARM = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=np.float64)
PREGRASP_ARM = np.array([0.0, 0.32, 0.0, -2.05, 0.0, 2.38, -0.7853], dtype=np.float64)
GRASP_ARM = np.array([0.0, 0.42, 0.0, -2.22, 0.0, 2.60, -0.7853], dtype=np.float64)
LIFT_ARM = np.array([0.0, 0.18, 0.0, -1.78, 0.0, 2.25, -0.7853], dtype=np.float64)
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 20.0


def scripted_targets(sim_time: float) -> tuple[np.ndarray, float]:
    if sim_time < 2.0:
        return HOME_ARM, GRIPPER_OPEN
    if sim_time < 4.0:
        return PREGRASP_ARM, GRIPPER_OPEN
    if sim_time < 5.5:
        return GRASP_ARM, GRIPPER_OPEN
    if sim_time < 7.0:
        return GRASP_ARM, GRIPPER_CLOSED
    return LIFT_ARM, GRIPPER_CLOSED


def run(policy: str = "scripted", steps: int = 1200, render_mode: str | None = "human") -> None:
    env = PandaCubeGraspEnv(render_mode=render_mode)

    try:
        _, info = env.reset(seed=0)
        relative_offset = info["cube_position"] - info["hand_position"]
        print(
            "reset:\n"
            f"  cube position = {np.round(info['cube_position'], 4)}\n"
            f"  hand position = {np.round(info['hand_position'], 4)}\n"
            f"  relative offset (cube - hand) = {np.round(relative_offset, 4)}"
        )

        for _ in range(steps):
            if policy == "scripted":
                arm_targets, gripper_target = scripted_targets(env.time)
                action = env.target_action(arm_targets, gripper_target)
            else:
                action = env.action_space.sample()

            _, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                status = "success" if terminated else "truncated"
                print(
                    status,
                    f"time={env.time:.2f}",
                    f"reward={reward:.3f}",
                    f"cube_height={info['cube_height']:.3f}",
                )
                env.reset()
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Panda cube grasp environment.")
    parser.add_argument("--policy", choices=("scripted", "random"), default="scripted")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--headless", action="store_true", help="Disable the MuJoCo viewer.")
    args = parser.parse_args()

    render_mode = None if args.headless else "human"
    run(policy=args.policy, steps=args.steps, render_mode=render_mode)


if __name__ == "__main__":
    main()
