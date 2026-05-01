from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_ARENA_PATH = ROOT_DIR / "panda_cube_grasp" / "assets" / "custom_grasp_arena.xml"

try:
    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
    from robosuite.models.arenas import Arena
    from robosuite.models.objects import BoxObject
    from robosuite.models.tasks import ManipulationTask
    from robosuite.controllers import load_composite_controller_config
    from robosuite.utils.mjcf_utils import CustomMaterial
    from robosuite.utils.observables import Observable, sensor
    from robosuite.utils.placement_samplers import UniformRandomSampler
    from robosuite.utils.transform_utils import convert_quat
except ModuleNotFoundError as exc:
    if exc.name == "robosuite":
        raise ModuleNotFoundError(
            "当前 Python 环境没有安装 robosuite。请先在运行 panda_grasp_demo.py 的同一个环境里安装："
            " pip install robosuite"
        ) from exc
    raise


SUCCESS_LIFT_HEIGHT = 0.08
GRASP_DISTANCE_THRESHOLD = 0.08
GRASP_OPENING_THRESHOLD = 0.06
PROJECT_TABLE_OFFSET = np.array((0.0, 0.0, 0.0), dtype=np.float64)
PROJECT_CUBE_XY = np.array((0.62, 0.18), dtype=np.float64)


class ProjectArena(Arena):
    """Arena matching this repository's MuJoCo scene style."""

    def __init__(self) -> None:
        super().__init__(str(PROJECT_ARENA_PATH))
        self.table_offset = PROJECT_TABLE_OFFSET.copy()
        self.table_full_size = np.array((0.9, 0.7, 0.05), dtype=np.float64)
        self.table_half_size = self.table_full_size / 2.0

    @property
    def table_top_abs(self) -> np.ndarray:
        return self.table_offset.copy()


class ProjectPandaCubeEnv(ManipulationEnv):
    """Project-specific robosuite task using Panda + a single red cube.

    This uses robosuite's manipulation base class for robot / gripper /
    controller plumbing, while keeping the arena, cube placement, reward, and
    success condition local to this project.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        base_types="default",
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.9, 0.9, 0.05),
        table_friction=(1.2, 0.03, 0.003),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ) -> None:
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = PROJECT_TABLE_OFFSET.copy()

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        if placement_initializer is None:
            placement_initializer = UniformRandomSampler(
                name="ProjectCubeSampler",
                x_range=(PROJECT_CUBE_XY[0], PROJECT_CUBE_XY[0]),
                y_range=(PROJECT_CUBE_XY[1], PROJECT_CUBE_XY[1]),
                rotation=0.0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=PROJECT_TABLE_OFFSET,
                z_offset=0.0,
            )
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def _load_model(self):
        super()._load_model()

        self.robots[0].robot_model.set_base_xpos([0.0, 0.0, 0.0])
        mujoco_arena = ProjectArena()
        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {"type": "cube"}
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size=[0.03, 0.03, 0.03],
            rgba=[1, 0, 0, 1],
            friction=[1.2, 0.03, 0.003],
            material=redwood,
            rng=self.rng,
        )

        self.placement_initializer.reset()
        self.placement_initializer.add_objects(self.cube)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            sensors = [cube_pos, cube_quat]
            observables.update(
                {
                    observable.__name__: Observable(
                        name=observable.__name__,
                        sensor=observable,
                        sampling_rate=self.control_freq,
                    )
                    for observable in sensors
                }
            )

        return observables

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def reward(self, action=None):
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        eef_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])
        distance_to_cube = np.linalg.norm(cube_pos - eef_pos)
        cube_height = cube_pos[2] - self.model.mujoco_arena.table_offset[2]
        is_success = self._check_success()

        if is_success:
            reward = 2.25
        elif self.reward_shaping:
            reach_reward = 1.0 - np.tanh(10.0 * distance_to_cube)
            grasp_reward = 0.25 if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube) else 0.0
            lift_reward = float(np.clip(cube_height / SUCCESS_LIFT_HEIGHT, 0.0, 1.0))
            reward = reach_reward + grasp_reward + lift_reward
        else:
            reward = 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25
        return reward

    def _check_success(self):
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]
        return bool(cube_height > table_height + SUCCESS_LIFT_HEIGHT)

    def _check_robot_configuration(self, robots):
        if isinstance(robots, (list, tuple)) and len(robots) != 1:
            raise ValueError("ProjectPandaCubeEnv expects exactly one robot.")


class RobosuitePandaCubeGraspEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium wrapper around robosuite ManipulationEnv with Panda + OSC_POSE.

    This keeps the project-facing API close to PandaCubeGraspEnv while delegating the
    robot model, torque actuators, gripper, and OSC controller to robosuite.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        reward_shaping: bool = True,
        control_freq: int = 20,
        controller_mode: str = "osc_pose",
        image_width: int = 640,
        image_height: int = 480,
    ) -> None:
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if controller_mode not in {"osc_position", "osc_pose"}:
            raise ValueError(f"Unsupported controller_mode: {controller_mode}")

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.reward_shaping = reward_shaping
        self.control_freq = control_freq
        self.controller_mode = controller_mode
        self.image_width = image_width
        self.image_height = image_height
        self._episode_step = 0

        self._rs_env = ProjectPandaCubeEnv(
            robots="Panda",
            base_types="NullMount",
            controller_configs=self._controller_config(),
            has_renderer=render_mode == "human",
            has_offscreen_renderer=render_mode == "rgb_array",
            render_camera=None if render_mode == "human" else "frontview",
            renderer="mjviewer",
            renderer_config={
                "cam_config": {
                    "lookat": [0.45, 0.0, 0.35],
                    "distance": 1.4,
                    "azimuth": 145,
                    "elevation": -22,
                }
            },
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=reward_shaping,
            control_freq=control_freq,
            horizon=max_episode_steps,
            ignore_done=True,
            camera_heights=image_height,
            camera_widths=image_width,
        )

        low, high = self._rs_env.action_spec
        self.action_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

        sample_obs = self._convert_obs(self._rs_env.reset())
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float64,
        )
        self._base_body_id = self._rs_env.sim.model.body_name2id("robot0_base")

    @property
    def time(self) -> float:
        return float(getattr(self._rs_env, "timestep", 0) / self.control_freq)

    @property
    def unwrapped_robosuite_env(self):
        return self._rs_env

    def is_running(self) -> bool:
        viewer = getattr(self._rs_env, "viewer", None)
        inner_viewer = getattr(viewer, "viewer", None)
        if inner_viewer is not None and hasattr(inner_viewer, "is_running"):
            return bool(inner_viewer.is_running())
        return True

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        self._episode_step = 0
        rs_obs = self._rs_env.reset()
        obs = self._convert_obs(rs_obs)
        info = self._get_info(rs_obs)

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        rs_obs, reward, terminated, rs_info = self._rs_env.step(action)
        self._episode_step += 1

        obs = self._convert_obs(rs_obs)
        info = self._get_info(rs_obs)
        info.update(rs_info)
        truncated = self._episode_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), bool(terminated), truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            self._rs_env.render()
            self._disable_viewer_shadows()
            return None
        if self.render_mode == "rgb_array":
            return self._rs_env.sim.render(
                camera_name="frontview",
                width=self.image_width,
                height=self.image_height,
            )[::-1]
        return None

    def _disable_viewer_shadows(self) -> None:
        viewer_renderer = getattr(self._rs_env, "viewer", None)
        viewer = getattr(viewer_renderer, "viewer", None)
        if viewer is None:
            return
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

    def close(self) -> None:
        self._rs_env.close()

    def _controller_config(self) -> dict:
        config = load_composite_controller_config(controller="BASIC", robot="Panda")
        right = config["body_parts"]["right"]
        if self.controller_mode == "osc_position":
            right["type"] = "OSC_POSITION"
            right["output_max"] = [0.05, 0.05, 0.05]
            right["output_min"] = [-0.05, -0.05, -0.05]
            right.pop("orientation_limits", None)
            right.pop("uncouple_pos_ori", None)
        else:
            right["type"] = "OSC_POSE"
            right["output_max"] = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
            right["output_min"] = [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]
            right["uncouple_pos_ori"] = True

        right["kp"] = 150
        right["damping_ratio"] = 1
        right["input_type"] = "delta"
        right["input_ref_frame"] = "base"
        config["body_parts"] = {"right": right}
        return config

    def _convert_obs(self, rs_obs: dict) -> np.ndarray:
        joint_pos = np.asarray(rs_obs["robot0_joint_pos"], dtype=np.float64)
        gripper_qpos = np.asarray(rs_obs["robot0_gripper_qpos"], dtype=np.float64)
        cube_pos = np.asarray(rs_obs["cube_pos"], dtype=np.float64)
        cube_quat = np.asarray(rs_obs["cube_quat"], dtype=np.float64)

        joint_vel = np.asarray(rs_obs["robot0_joint_vel"], dtype=np.float64)
        gripper_qvel = np.asarray(rs_obs["robot0_gripper_qvel"], dtype=np.float64)
        cube_vel = np.zeros(6, dtype=np.float64)

        hand_pos = np.asarray(rs_obs["robot0_eef_pos"], dtype=np.float64)
        relative_pos = cube_pos - hand_pos
        gripper_opening = np.array([float(np.sum(gripper_qpos))], dtype=np.float64)

        return np.concatenate(
            [
                joint_pos,
                gripper_qpos,
                cube_pos,
                cube_quat,
                joint_vel,
                gripper_qvel,
                cube_vel,
                hand_pos,
                cube_pos,
                relative_pos,
                gripper_opening,
            ]
        )

    def _get_info(self, rs_obs: dict) -> dict:
        cube_pos = np.asarray(rs_obs["cube_pos"], dtype=np.float64).copy()
        hand_pos = np.asarray(rs_obs["robot0_eef_pos"], dtype=np.float64).copy()
        base_pos = np.asarray(self._rs_env.sim.data.body_xpos[self._base_body_id], dtype=np.float64).copy()
        base_rot = np.asarray(self._rs_env.sim.data.body_xmat[self._base_body_id], dtype=np.float64).reshape(3, 3).copy()
        world_to_base = base_rot.T
        cube_pos_base = world_to_base @ (cube_pos - base_pos)
        hand_pos_base = world_to_base @ (hand_pos - base_pos)
        gripper_qpos = np.asarray(rs_obs["robot0_gripper_qpos"], dtype=np.float64)
        distance_to_cube = float(np.linalg.norm(cube_pos - hand_pos))
        table_height = float(self._rs_env.model.mujoco_arena.table_offset[2])
        cube_height = float(cube_pos[2] - table_height)
        gripper_opening = float(np.sum(gripper_qpos))
        is_grasping = distance_to_cube < GRASP_DISTANCE_THRESHOLD and gripper_opening < GRASP_OPENING_THRESHOLD

        is_success = False
        if hasattr(self._rs_env, "_check_success"):
            is_success = bool(self._rs_env._check_success())
        else:
            is_success = cube_height > SUCCESS_LIFT_HEIGHT

        return {
            "cube_position": cube_pos_base,
            "hand_position": hand_pos_base,
            "finger_midpoint": hand_pos_base,
            "cube_position_world": cube_pos,
            "hand_position_world": hand_pos,
            "robot_base_position_world": base_pos,
            "distance_to_cube": distance_to_cube,
            "cube_height": cube_height,
            "gripper_opening": gripper_opening,
            "is_grasping": is_grasping,
            "is_success": is_success,
        }
