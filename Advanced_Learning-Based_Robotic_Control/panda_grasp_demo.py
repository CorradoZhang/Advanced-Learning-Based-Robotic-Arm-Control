from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
ARENA_XML = ROOT_DIR / "panda_cube_grasp" / "assets" / "custom_grasp_arena.xml"


def pose_action(position_action: np.ndarray, rotation_action: np.ndarray | None = None, gripper: float = 0.0) -> np.ndarray:
    if rotation_action is None:
        rotation_action = np.zeros(3, dtype=np.float64)
    return np.concatenate(
        [
            np.asarray(position_action, dtype=np.float64),
            np.asarray(rotation_action, dtype=np.float64),
            np.array([gripper], dtype=np.float64),
        ]
    ).astype(np.float32)


def print_scene_summary(env, info: dict) -> None:
    rs_env = env.unwrapped_robosuite_env
    model = rs_env.sim.model
    data = rs_env.sim.data

    geom_names = [model.geom_id2name(i) for i in range(model.ngeom)]
    body_names = [model.body_id2name(i) for i in range(model.nbody)]
    camera_names = [model.camera_id2name(i) for i in range(model.ncam)]

    workspace_geoms = [name for name in geom_names if name and name.startswith("workspace")]
    mount_geoms = [
        name
        for name in geom_names
        if name and ("pedestal" in name or "controller_box" in name or "torso" in name)
    ]

    print("Scene check")
    print(f"  arena xml          = {ARENA_XML}")
    print(f"  controller         = {env.controller_mode}")
    print(f"  action space       = shape={env.action_space.shape}, low={env.action_space.low}, high={env.action_space.high}")
    print(f"  cameras            = {camera_names}")
    print(f"  workspace geoms    = {workspace_geoms}")
    print(f"  robosuite mounts   = {mount_geoms if mount_geoms else 'none'}")
    print(f"  robot bodies       = {[name for name in body_names if name and name.startswith('robot0_')][:8]} ...")
    print(f"  robot base world   = {np.round(info['robot_base_position_world'], 4)}")
    print(f"  hand base frame    = {np.round(info['hand_position'], 4)}")
    print(f"  hand world frame   = {np.round(info['hand_position_world'], 4)}")
    print(f"  cube base frame    = {np.round(info['cube_position'], 4)}")
    print(f"  cube world frame   = {np.round(info['cube_position_world'], 4)}")
    print(f"  qpos sample        = {np.round(data.qpos[:7], 4)}")


def circle_action(step: int, rate: float, radius_action: float) -> np.ndarray:
    phase = 2.0 * np.pi * (step / rate) / 6.0
    position = radius_action * np.array([np.cos(phase), np.sin(phase), 0.25 * np.sin(2.0 * phase)])
    return pose_action(position)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open and sanity-check the custom robosuite Panda grasp scene.")
    parser.add_argument("--mode", choices=("hold", "circle", "static"), default="circle")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--status-every", type=float, default=1.0)
    parser.add_argument("--circle-action", type=float, default=0.45)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    try:
        from panda_cube_grasp import RobosuitePandaCubeGraspEnv
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    env = RobosuitePandaCubeGraspEnv(
        render_mode=None if args.headless else "human",
        max_episode_steps=1_000_000,
        control_freq=int(args.rate),
        controller_mode="osc_pose",
    )

    try:
        _, info = env.reset(seed=0)
        print_scene_summary(env, info)

        if args.mode == "static":
            if not args.headless:
                print("Static scene is open. Close the MuJoCo viewer or press Ctrl+C to quit.")
                while env.is_running():
                    env.render()
                    time.sleep(1.0 / args.rate)
            return

        print(f"Running scene test mode: {args.mode}")
        print("Default mode is circle, so the arm should move visibly. Use --mode static to only inspect the scene.")

        step = 0
        next_status = time.monotonic()
        dt = 1.0 / args.rate

        while env.is_running():
            if args.max_steps is not None and step >= args.max_steps:
                break

            loop_start = time.monotonic()
            if args.mode == "circle":
                action = circle_action(step, args.rate, args.circle_action)
            else:
                action = pose_action(np.zeros(3, dtype=np.float64))

            _, _, _, _, info = env.step(action)
            step += 1

            now = time.monotonic()
            if now >= next_status:
                print(
                    f"step={step:05d}",
                    f"hand={np.round(info['hand_position'], 4)}",
                    f"cube={np.round(info['cube_position'], 4)}",
                    f"dist={info['distance_to_cube']:.3f}",
                )
                next_status = now + args.status_every

            sleep_time = dt - (time.monotonic() - loop_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
    except RuntimeError as exc:
        if "launch_passive" in str(exc) and "mjpython" in str(exc):
            raise SystemExit("macOS 上打开 MuJoCo viewer 需要用：mjpython panda_grasp_demo.py") from exc
        raise
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
