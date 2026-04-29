from __future__ import annotations

"""MuJoCo Panda cube-grasp environments.

This file contains two closely related environments:

- PandaCubeGraspEnv:
  Joint-space control. The policy outputs 7 arm joint increments plus 1 gripper increment.
- PandaCubeGraspIKEnv:
  Task-space control. The policy outputs xyz hand motion, xyz orientation motion, plus
  1 gripper increment, and a Jacobian-based IK step converts the task-space delta into
  arm joint updates.

Both environments share the same scene, observation layout, reward design, and rendering code.
"""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "mujoco_menagerie" / "franka_emika_panda" / "grasp_scene.xml"

HOME_ARM = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=np.float64)
HOME_CTRL = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 255.0], dtype=np.float64)
DEFAULT_CUBE_POSE = np.array([0.62, 0.18, 0.03, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
GROUND_SURFACE_Z = 0.0
SUCCESS_LIFT_HEIGHT = 0.08
REACH_REWARD_GAIN = 10.0
GRASP_DISTANCE_THRESHOLD = 0.08
GRASP_OPENING_THRESHOLD = 0.06
GRASP_REWARD = 0.25
SUCCESS_REWARD = 2.25
IK_POSITION_SCALE = np.array([0.025, 0.025, 0.025], dtype=np.float64)
IK_ORIENTATION_SCALE = np.array([0.12, 0.12, 0.12], dtype=np.float64)
IK_GRIPPER_SCALE = 25.0
IK_DAMPING = 1.0e-4
IK_MAX_DQ_NORM = 0.25
IK_TASK_WEIGHTS = np.array([1.0, 1.0, 1.0, 0.35, 0.35, 0.35], dtype=np.float64)
IK_MAX_JOINT_STEP = np.array([0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.04], dtype=np.float64)
IK_BASE_DAMPING = 1.0e-4
IK_ADAPTIVE_DAMPING_GAIN = 5.0e-4
IK_POSITION_GAIN = 1.0
IK_ORIENTATION_GAIN = 0.6
IK_MAX_POSITION_ERROR = 0.05
IK_MAX_ORIENTATION_ERROR = 0.25
IK_SOLVER_ITERATIONS = 8
IK_SOLVER_STEP_SIZE = 0.8
IK_POSITION_TOLERANCE = 1.0e-3
IK_ORIENTATION_TOLERANCE = 2.0e-3
DEFAULT_CAMERA_LOOKAT = np.array([0.55, 0.08, 0.18], dtype=np.float64)
DEFAULT_CAMERA_DISTANCE = 3
DEFAULT_CAMERA_AZIMUTH = 145
DEFAULT_CAMERA_ELEVATION = -22


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def _so3_exp(rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec)
    if theta < 1.0e-8:
        return np.eye(3, dtype=np.float64) + _skew(rotvec)

    axis = rotvec / theta
    K = _skew(axis)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _rotation_error_world(current_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    # Small-angle orientation error expressed in the world frame, consistent with mj_jacBody.
    return 0.5 * (
        np.cross(current_rot[:, 0], target_rot[:, 0])
        + np.cross(current_rot[:, 1], target_rot[:, 1])
        + np.cross(current_rot[:, 2], target_rot[:, 2])
    )


def _named_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id == -1:
        raise ValueError(f"Required MuJoCo object not found: {name}")
    return obj_id


class PandaCubeGraspEnv(gym.Env[np.ndarray, np.ndarray]):
    """State-based Panda grasp environment with direct joint-space actions."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 5,
        max_episode_steps: int = 200,
        cube_xy_jitter: float = 0.0,
        image_width: int = 640,
        image_height: int = 480,
    ) -> None:
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.cube_xy_jitter = cube_xy_jitter
        self.image_width = image_width
        self.image_height = image_height

        # Load the MuJoCo model once and cache ids / addresses that are needed every step.
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self._viewer = None
        self._rgb_renderer = None
        self._render_camera = self._build_render_camera()

        self._cube_joint_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        self._cube_qpos_adr = self.model.jnt_qposadr[self._cube_joint_id]
        self._cube_dof_adr = self.model.jnt_dofadr[self._cube_joint_id]

        self._hand_body_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._left_finger_body_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self._right_finger_body_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self._cube_body_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._cube_geom_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        arm_joint_names = tuple(f"joint{i}" for i in range(1, 8))
        self._arm_joint_ids = [
            _named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in arm_joint_names
        ]
        self._arm_qpos_adrs = [self.model.jnt_qposadr[joint_id] for joint_id in self._arm_joint_ids]
        self._arm_dof_adrs = [self.model.jnt_dofadr[joint_id] for joint_id in self._arm_joint_ids]
        self._arm_joint_low = np.array([self.model.jnt_range[joint_id][0] for joint_id in self._arm_joint_ids])
        self._arm_joint_high = np.array([self.model.jnt_range[joint_id][1] for joint_id in self._arm_joint_ids])

        finger_joint_names = ("finger_joint1", "finger_joint2")
        self._finger_joint_ids = [
            _named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in finger_joint_names
        ]
        self._finger_qpos_adrs = [self.model.jnt_qposadr[joint_id] for joint_id in self._finger_joint_ids]

        # The agent controls actuator targets rather than writing qpos directly.
        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._action_scale = np.array([0.08, 0.08, 0.08, 0.08, 0.10, 0.10, 0.10, 25.0], dtype=np.float64)
        self._ctrl_target = HOME_CTRL.copy()
        self._episode_step = 0
        self._cube_half_size = self.model.geom_size[self._cube_geom_id].copy()
        self._cube_spawn_xy_min = np.array([0.20, -0.30], dtype=np.float64)
        self._cube_spawn_xy_max = np.array([0.80, 0.30], dtype=np.float64)
        self._cube_spawn_z = float(GROUND_SURFACE_Z + self._cube_half_size[2])

        self._reset_simulation_state()
        obs = self._get_obs()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float64,
        )

    @property
    def time(self) -> float:
        return float(self.data.time)

    def is_running(self) -> bool:
        if self.render_mode != "human":
            return True
        if self._viewer is None:
            return True
        return self._viewer.is_running()

    def target_action(self, arm_targets: np.ndarray | list[float], gripper_target: float) -> np.ndarray:
        desired_ctrl = np.concatenate(
            [np.asarray(arm_targets, dtype=np.float64), np.array([gripper_target], dtype=np.float64)]
        )
        action = (desired_ctrl - self._ctrl_target) / self._action_scale
        return np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._episode_step = 0
        self._reset_simulation_state()
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Interpret the action as an increment on the current actuator targets.
        self._ctrl_target = np.clip(
            self._ctrl_target + action * self._action_scale,
            self._ctrl_low,
            self._ctrl_high,
        )

        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self._ctrl_target
            mujoco.mj_step(self.model, self.data)

        self._episode_step += 1

        obs = self._get_obs()
        info = self._get_info()
        reward = self._get_reward(action, info)
        terminated = bool(info["is_success"])
        truncated = self._episode_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            if self._viewer is None or not self._viewer.is_running():
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer.cam.lookat[:] = DEFAULT_CAMERA_LOOKAT
                self._viewer.cam.distance = DEFAULT_CAMERA_DISTANCE
                self._viewer.cam.azimuth = DEFAULT_CAMERA_AZIMUTH
                self._viewer.cam.elevation = DEFAULT_CAMERA_ELEVATION

            self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._rgb_renderer is None:
                self._rgb_renderer = mujoco.Renderer(
                    self.model,
                    height=self.image_height,
                    width=self.image_width,
                )

            self._rgb_renderer.update_scene(self.data, camera=self._render_camera)
            return self._rgb_renderer.render()

        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._rgb_renderer is not None:
            self._rgb_renderer.close()
            self._rgb_renderer = None

    def _reset_simulation_state(self) -> None:
        # Reset the robot to the home keyframe, then overwrite the cube freejoint pose.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._ctrl_target = HOME_CTRL.copy()
        self.data.ctrl[:] = self._ctrl_target
        self._reset_cube_pose()
        mujoco.mj_forward(self.model, self.data)

    def _reset_cube_pose(self) -> None:
        cube_pose = DEFAULT_CUBE_POSE.copy()
        if self.cube_xy_jitter > 0.0:
            cube_pose[:2] += self.np_random.uniform(-self.cube_xy_jitter, self.cube_xy_jitter, size=2)

        # Keep the cube center inside the tabletop support polygon and place it exactly on the surface.
        cube_pose[:2] = np.clip(cube_pose[:2], self._cube_spawn_xy_min, self._cube_spawn_xy_max)
        cube_pose[2] = self._cube_spawn_z

        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 7] = cube_pose
        self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 6] = 0.0

    def _get_obs(self) -> np.ndarray:
        # Observations stay in low-level state space even for the IK variant so training
        # comparisons focus on the action parameterization rather than perception changes.
        hand_pos = self.data.xpos[self._hand_body_id]
        cube_pos = self.data.xpos[self._cube_body_id]
        finger_midpoint = self._finger_midpoint()
        gripper_opening = self._gripper_opening()

        return np.concatenate(
            [
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                hand_pos.copy(),
                cube_pos.copy(),
                (cube_pos - finger_midpoint).copy(),
                np.array([gripper_opening], dtype=np.float64),
            ]
        )

    def _get_info(self) -> dict:
        cube_pos = self.data.xpos[self._cube_body_id].copy()
        hand_pos = self.data.xpos[self._hand_body_id].copy()
        finger_midpoint = self._finger_midpoint()
        distance_to_cube = float(np.linalg.norm(cube_pos - finger_midpoint))
        cube_height = float(cube_pos[2] - GROUND_SURFACE_Z)
        is_success = cube_height > SUCCESS_LIFT_HEIGHT
        gripper_opening = self._gripper_opening()
        is_grasping = self._is_grasping(distance_to_cube, gripper_opening)

        return {
            "cube_position": cube_pos,
            "hand_position": hand_pos,
            "finger_midpoint": finger_midpoint,
            "distance_to_cube": distance_to_cube,
            "cube_height": cube_height,
            "gripper_opening": gripper_opening,
            "is_grasping": is_grasping,
            "is_success": is_success,
        }

    def _get_reward(self, action: np.ndarray, info: dict) -> float:
        # Robosuite-style staged shaping:
        # 1. reach the cube
        # 2. close around it
        # 3. lift it off the table
        # 4. give a fixed terminal bonus on success
        reach_reward = 1.0 - np.tanh(REACH_REWARD_GAIN * info["distance_to_cube"])
        grasp_reward = GRASP_REWARD if info["is_grasping"] else 0.0
        lift_reward = float(np.clip(info["cube_height"] / SUCCESS_LIFT_HEIGHT, 0.0, 1.0))
        success_reward = SUCCESS_REWARD if info["is_success"] else 0.0
        action_penalty = 0.01 * float(np.linalg.norm(action))

        shaped_reward = reach_reward + grasp_reward + lift_reward
        reward = success_reward if info["is_success"] else shaped_reward

        return reward - action_penalty

    def _finger_midpoint(self) -> np.ndarray:
        left_pos = self.data.xpos[self._left_finger_body_id]
        right_pos = self.data.xpos[self._right_finger_body_id]
        return 0.5 * (left_pos + right_pos)

    def _gripper_opening(self) -> float:
        return float(sum(self.data.qpos[qpos_adr] for qpos_adr in self._finger_qpos_adrs))

    def _is_grasping(self, distance_to_cube: float, gripper_opening: float) -> bool:
        return distance_to_cube < GRASP_DISTANCE_THRESHOLD and gripper_opening < GRASP_OPENING_THRESHOLD

    def _build_render_camera(self) -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = DEFAULT_CAMERA_LOOKAT
        camera.distance = DEFAULT_CAMERA_DISTANCE
        camera.azimuth = DEFAULT_CAMERA_AZIMUTH
        camera.elevation = DEFAULT_CAMERA_ELEVATION
        return camera

class PandaCubeGraspIKEnv(PandaCubeGraspEnv):
    """Task-space Panda grasp environment using a Jacobian-based damped least-squares IK controller."""

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 5,
        max_episode_steps: int = 200,
        cube_xy_jitter: float = 0.0,
        image_width: int = 640,
        image_height: int = 480,
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            cube_xy_jitter=cube_xy_jitter,
            image_width=image_width,
            image_height=image_height,
        )
        # Task-space policy: xyz position delta, xyz orientation delta, plus 1 gripper command.
        self._ik_position_scale = IK_POSITION_SCALE.copy()
        self._ik_orientation_scale = IK_ORIENTATION_SCALE.copy()
        self._ik_gripper_scale = IK_GRIPPER_SCALE
        self._ik_task_weights = IK_TASK_WEIGHTS.copy()
        self._ik_max_joint_step = IK_MAX_JOINT_STEP.copy()
        self._ik_joint_vel_limits = None
        self._ik_base_damping = IK_BASE_DAMPING
        self._ik_adaptive_damping_gain = IK_ADAPTIVE_DAMPING_GAIN
        self._ik_pos_gain = IK_POSITION_GAIN
        self._ik_rot_gain = IK_ORIENTATION_GAIN
        self._ik_max_position_error = IK_MAX_POSITION_ERROR
        self._ik_max_orientation_error = IK_MAX_ORIENTATION_ERROR
        self._ik_solver_iterations = IK_SOLVER_ITERATIONS
        self._ik_solver_step_size = IK_SOLVER_STEP_SIZE
        self._ik_position_tolerance = IK_POSITION_TOLERANCE
        self._ik_orientation_tolerance = IK_ORIENTATION_TOLERANCE
        self._ik_data = mujoco.MjData(self.model)
        self._ik_target_pos: np.ndarray | None = None
        self._ik_target_rot: np.ndarray | None = None
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self._reset_ik_targets()

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        obs, info = super().reset(seed=seed, options=options)
        self._reset_ik_targets()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Interpret the policy output in task space: xyz position deltas, xyz orientation deltas,
        # and one gripper command.
        delta_pos = action[:3] * self._ik_position_scale
        delta_rot = action[3:6] * self._ik_orientation_scale
        q_des = self._solve_pose_ik(delta_pos, delta_rot)
        self._ctrl_target[:7] = np.clip(
            q_des,
            self._ctrl_low[:7],
            self._ctrl_high[:7],
        )
        self._ctrl_target[7] = np.clip(
            self._ctrl_target[7] + action[6] * self._ik_gripper_scale,
            self._ctrl_low[7],
            self._ctrl_high[7],
        )

        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self._ctrl_target
            mujoco.mj_step(self.model, self.data)

        self._episode_step += 1

        obs = self._get_obs()
        info = self._get_info()
        reward = self._get_reward(action, info)
        terminated = bool(info["is_success"])
        truncated = self._episode_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _reset_ik_targets(self) -> None:
        self._ik_target_pos = self._finger_midpoint().copy()
        self._ik_target_rot = self.data.xmat[self._hand_body_id].reshape(3, 3).copy()

    def _ik_position_and_jacobian(self, ik_data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        # Use the fingertip midpoint as the Cartesian control point for position.
        # This is more relevant to grasping than the hand body origin.
        left_pos = ik_data.xpos[self._left_finger_body_id].copy()
        right_pos = ik_data.xpos[self._right_finger_body_id].copy()
        control_pos = 0.5 * (left_pos + right_pos)

        left_jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        left_jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        right_jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        right_jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, ik_data, left_jacp, left_jacr, self._left_finger_body_id)
        mujoco.mj_jacBody(self.model, ik_data, right_jacp, right_jacr, self._right_finger_body_id)

        control_jacp = 0.5 * (left_jacp + right_jacp)
        return control_pos, control_jacp[:, self._arm_dof_adrs]

    def _solve_pose_ik(self, delta_pos: np.ndarray, delta_rot: np.ndarray) -> np.ndarray:
        # Solve a small local IK problem each control step, following the iterative
        # Jacobian-based Levenberg-Marquardt / damped least-squares pattern.
        if self._ik_target_pos is None or self._ik_target_rot is None:
            self._reset_ik_targets()

        self._ik_target_pos = self._ik_target_pos + delta_pos
        self._ik_target_rot = _so3_exp(delta_rot) @ self._ik_target_rot

        q_current = self.data.qpos[self._arm_qpos_adrs].copy()
        self._ik_data.qpos[:] = self.data.qpos
        self._ik_data.qvel[:] = self.data.qvel
        mujoco.mj_forward(self.model, self._ik_data)

        target_pos = self._ik_target_pos.copy()
        target_rot = self._ik_target_rot.copy()
        q_work = self._ctrl_target[:7].copy()

        for _ in range(self._ik_solver_iterations):
            self._ik_data.qpos[:] = self.data.qpos
            self._ik_data.qvel[:] = self.data.qvel
            self._ik_data.qpos[self._arm_qpos_adrs] = q_work
            mujoco.mj_forward(self.model, self._ik_data)

            current_pos, control_jacobian = self._ik_position_and_jacobian(self._ik_data)
            current_rot = self._ik_data.xmat[self._hand_body_id].reshape(3, 3).copy()

            pos_err = target_pos - current_pos
            rot_err = _rotation_error_world(current_rot, target_rot)

            pos_err_norm = np.linalg.norm(pos_err)
            if pos_err_norm > self._ik_max_position_error:
                pos_err = pos_err * (self._ik_max_position_error / pos_err_norm)

            rot_err_norm = np.linalg.norm(rot_err)
            if rot_err_norm > self._ik_max_orientation_error:
                rot_err = rot_err * (self._ik_max_orientation_error / rot_err_norm)

            if pos_err_norm < self._ik_position_tolerance and rot_err_norm < self._ik_orientation_tolerance:
                break

            task_err = np.concatenate([self._ik_pos_gain * pos_err, self._ik_rot_gain * rot_err])

            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            jacp_dummy = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.model, self._ik_data, jacp_dummy, jacr, self._hand_body_id)

            arm_jacobian = np.vstack(
                [
                    control_jacobian,
                    jacr[:, self._arm_dof_adrs],
                ]
            )
            weight_matrix = np.diag(self._ik_task_weights)
            weighted_jacobian = weight_matrix @ arm_jacobian
            weighted_error = weight_matrix @ task_err

            singular_values = np.linalg.svd(weighted_jacobian, compute_uv=False)
            sigma_min = singular_values[-1] if singular_values.size else 0.0
            damping = self._ik_base_damping + self._ik_adaptive_damping_gain / (sigma_min + 1.0e-8)
            lambda_sq = damping * damping

            hessian = weighted_jacobian.T @ weighted_jacobian + lambda_sq * np.eye(weighted_jacobian.shape[1])
            gradient = weighted_jacobian.T @ weighted_error
            dq = np.linalg.solve(hessian, gradient)
            dq = self._ik_solver_step_size * dq
            dq = np.clip(dq, -self._ik_max_joint_step, self._ik_max_joint_step)

            q_work = np.clip(q_work + dq, self._arm_joint_low, self._arm_joint_high)

        if self._ik_joint_vel_limits is not None:
            dt = self.model.opt.timestep * self.frame_skip
            dq_total = q_work - q_current
            dq_vel_clip = self._ik_joint_vel_limits * dt
            dq_total = np.clip(dq_total, -dq_vel_clip, dq_vel_clip)
            q_work = q_current + dq_total

        return q_work
