from gymnasium.envs.registration import register, registry

if "PandaCubeGrasp-v0" not in registry:
    register(
        id="PandaCubeGrasp-v0",
        entry_point="panda_cube_grasp.envs:PandaCubeGraspEnv",
        max_episode_steps=200,
    )

if "PandaCubeGraspIK-v0" not in registry:
    register(
        id="PandaCubeGraspIK-v0",
        entry_point="panda_cube_grasp.envs:PandaCubeGraspIKEnv",
        max_episode_steps=200,
    )

if "PandaCubeGraspMocap-v0" not in registry:
    register(
        id="PandaCubeGraspMocap-v0",
        entry_point="panda_cube_grasp.envs:PandaCubeGraspMocapEnv",
        max_episode_steps=200,
    )

if "PandaCubeGraspOSCPosition-v0" not in registry:
    register(
        id="PandaCubeGraspOSCPosition-v0",
        entry_point="panda_cube_grasp.envs:PandaCubeGraspOSCPositionEnv",
        max_episode_steps=200,
    )

if "PandaCubeGraspRobosuite-v0" not in registry:
    register(
        id="PandaCubeGraspRobosuite-v0",
        entry_point="panda_cube_grasp.envs:RobosuitePandaCubeGraspEnv",
        max_episode_steps=200,
    )

from panda_cube_grasp.envs import (
    PandaCubeGraspEnv,
    PandaCubeGraspIKEnv,
    PandaCubeGraspMocapEnv,
    PandaCubeGraspOSCPositionEnv,
)

__all__ = [
    "PandaCubeGraspEnv",
    "PandaCubeGraspIKEnv",
    "PandaCubeGraspMocapEnv",
    "PandaCubeGraspOSCPositionEnv",
    "RobosuitePandaCubeGraspEnv",
]


def __getattr__(name: str):
    if name == "RobosuitePandaCubeGraspEnv":
        from panda_cube_grasp.envs.robosuite_panda_cube_grasp_env import RobosuitePandaCubeGraspEnv

        return RobosuitePandaCubeGraspEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
