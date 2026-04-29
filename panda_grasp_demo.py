from __future__ import annotations

import numpy as np

from panda_cube_grasp import PandaCubeGraspIKEnv


# This demo is intentionally a pure IK motion test. It avoids object contact so the
# viewer shows whether task-space control itself is smooth and stable.
POSITION_ACTION_SCALE = 0.05
ORIENTATION_ACTION_SCALE = 0.20
MAX_XY_ACTION = 0.10
MAX_Z_ACTION = 0.10
MAX_ORIENTATION_ACTION = 0.15
GRIPPER_ACTION = 0.0
WAYPOINT_TOLERANCE = 0.015
SQUARE_HALF_SPAN = 0.08


def _orientation_error(current_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    # Small-angle orientation error in world coordinates.
    return 0.5 * (
        np.cross(current_rot[:, 0], target_rot[:, 0])
        + np.cross(current_rot[:, 1], target_rot[:, 1])
        + np.cross(current_rot[:, 2], target_rot[:, 2])
    )


def _target_to_action(
    target_pos: np.ndarray,
    current_pos: np.ndarray,
    current_rot: np.ndarray,
    target_rot: np.ndarray,
) -> np.ndarray:
    # IK action format: [dx, dy, dz, droll, dpitch, dyaw, dgripper].
    position_error = target_pos - current_pos
    xyz_action = position_error / POSITION_ACTION_SCALE
    xyz_action[0] = np.clip(xyz_action[0], -MAX_XY_ACTION, MAX_XY_ACTION)
    xyz_action[1] = np.clip(xyz_action[1], -MAX_XY_ACTION, MAX_XY_ACTION)
    xyz_action[2] = np.clip(xyz_action[2], -MAX_Z_ACTION, MAX_Z_ACTION)
    orientation_error = _orientation_error(current_rot, target_rot)
    orientation_action = np.clip(
        orientation_error / ORIENTATION_ACTION_SCALE,
        -MAX_ORIENTATION_ACTION,
        MAX_ORIENTATION_ACTION,
    )
    return np.concatenate([xyz_action, orientation_action, np.array([GRIPPER_ACTION], dtype=np.float64)]).astype(
        np.float32
    )


def build_waypoints(start_pos: np.ndarray) -> list[np.ndarray]:
    # Use a simple horizontal square so the demo tests left / right / forward / backward
    # task-space motion without introducing height changes or contact.
    return [
        start_pos + np.array([-SQUARE_HALF_SPAN, -SQUARE_HALF_SPAN, 0.00], dtype=np.float64),
        start_pos + np.array([+SQUARE_HALF_SPAN, -SQUARE_HALF_SPAN, 0.00], dtype=np.float64),
        start_pos + np.array([+SQUARE_HALF_SPAN, +SQUARE_HALF_SPAN, 0.00], dtype=np.float64),
        start_pos + np.array([-SQUARE_HALF_SPAN, +SQUARE_HALF_SPAN, 0.00], dtype=np.float64),
    ]


def main() -> None:
    env = PandaCubeGraspIKEnv(
        render_mode="human",
        frame_skip=5,
        max_episode_steps=1_000_000,
    )

    try:
        _, info = env.reset(seed=0)
        start_pos = info["finger_midpoint"].copy()
        target_rot = env.data.xmat[env._hand_body_id].reshape(3, 3).copy()
        waypoints = build_waypoints(start_pos)
        waypoint_index = 0

        print(
            "reset:",
            f"cube={np.round(info['cube_position'], 4)}",
            f"finger_midpoint={np.round(start_pos, 4)}",
        )
        print("IK demo waypoints:")
        for idx, waypoint in enumerate(waypoints):
            print(f"  {idx}: {np.round(waypoint, 4)}")

        while env.is_running():
            current_pos = info["finger_midpoint"]
            current_rot = env.data.xmat[env._hand_body_id].reshape(3, 3)
            target_pos = waypoints[waypoint_index]

            if np.linalg.norm(target_pos - current_pos) < WAYPOINT_TOLERANCE:
                waypoint_index = (waypoint_index + 1) % len(waypoints)
                target_pos = waypoints[waypoint_index]

            action = _target_to_action(target_pos, current_pos, current_rot, target_rot)
            _, _, _, _, info = env.step(action)
    finally:
        env.close()


if __name__ == "__main__":
    main()
