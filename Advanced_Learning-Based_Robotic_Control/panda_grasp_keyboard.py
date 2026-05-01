from __future__ import annotations

import argparse
import time
from copy import deepcopy

import numpy as np


def print_status(info: dict) -> None:
    cube = np.round(info["cube_position"], 3)
    hand = np.round(info["hand_position"], 3)
    dist = info["distance_to_cube"]
    grip = info["gripper_opening"]
    print(f"hand={hand} cube={cube} dist={dist:.3f} grip={grip:.3f}")


def build_env(control_freq: int):
    from panda_cube_grasp import RobosuitePandaCubeGraspEnv

    return RobosuitePandaCubeGraspEnv(
        render_mode="human",
        max_episode_steps=1_000_000,
        control_freq=control_freq,
        controller_mode="osc_pose",
    )


def build_device(rs_env, pos_sensitivity: float, rot_sensitivity: float):
    from robosuite.devices import Keyboard

    device = Keyboard(
        env=rs_env,
        pos_sensitivity=pos_sensitivity,
        rot_sensitivity=rot_sensitivity,
    )
    return device


def create_flat_action(rs_env, device, input_action_dict: dict, prev_gripper_actions: dict[str, np.ndarray]) -> np.ndarray:
    active_robot = rs_env.robots[device.active_robot]
    action_dict = deepcopy(input_action_dict)

    for arm in active_robot.arms:
        if hasattr(active_robot, "composite_controller") and active_robot.composite_controller is not None:
            controller_input_type = active_robot.part_controllers[arm].input_type
        else:
            controller_input_type = active_robot.part_controllers[arm].input_type

        if controller_input_type == "delta":
            action_dict[arm] = input_action_dict[f"{arm}_delta"]
        elif controller_input_type == "absolute":
            action_dict[arm] = input_action_dict[f"{arm}_abs"]
        else:
            raise ValueError(f"Unsupported controller input type: {controller_input_type}")

    env_action = [robot.create_action_vector(prev_gripper_actions[i]) for i, robot in enumerate(rs_env.robots)]
    env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
    env_action = np.concatenate(env_action)

    for gripper_key in prev_gripper_actions[device.active_robot]:
        prev_gripper_actions[device.active_robot][gripper_key] = action_dict[gripper_key]

    return env_action.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="robosuite native keyboard teleoperation for Panda OSC_POSITION.")
    parser.add_argument("--rate", type=float, default=20.0, help="Control loop frequency in Hz.")
    parser.add_argument("--status-every", type=float, default=1.0, help="Seconds between status prints.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional maximum number of control steps.")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="robosuite keyboard position sensitivity.")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="robosuite keyboard rotation sensitivity.")
    args = parser.parse_args()

    env = build_env(control_freq=int(args.rate))
    rs_env = env.unwrapped_robosuite_env
    device = build_device(rs_env, args.pos_sensitivity, args.rot_sensitivity)

    try:
        _, info = env.reset(seed=0)
        env.render()
        device.start_control()

        prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in rs_env.robots
        ]

        print("Using robosuite native Keyboard device.")
        print("Important: click the MuJoCo window once before pressing keys.")
        print("On macOS, if you see an input monitoring warning, enable accessibility / input monitoring for mjpython or your terminal.")
        print(
            "reset:",
            f"cube={np.round(info['cube_position'], 4)}",
            f"hand={np.round(info['hand_position'], 4)}",
        )

        dt = 1.0 / args.rate
        next_status_time = time.monotonic()
        step_count = 0

        while env.is_running():
            if args.max_steps is not None and step_count >= args.max_steps:
                break

            loop_start = time.monotonic()

            input_action_dict = device.input2action()

            # robosuite uses reset-from-device semantics. For the keyboard this is
            # triggered by Ctrl+q, so we mirror the official demo behavior here.
            if input_action_dict is None:
                _, info = env.reset()
                env.render()
                device.start_control()
                prev_gripper_actions = [
                    {
                        f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                        for robot_arm in robot.arms
                        if robot.gripper[robot_arm].dof > 0
                    }
                    for robot in rs_env.robots
                ]
                continue

            flat_action = create_flat_action(rs_env, device, input_action_dict, prev_gripper_actions)
            _, _, _, _, info = env.step(flat_action)
            step_count += 1

            now = time.monotonic()
            if now >= next_status_time:
                print_status(info)
                next_status_time = now + args.status_every

            sleep_time = dt - (time.monotonic() - loop_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    except RuntimeError as exc:
        if "launch_passive" in str(exc) and "mjpython" in str(exc):
            raise SystemExit("On macOS, run this script with: mjpython panda_grasp_keyboard.py") from exc
        raise
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
