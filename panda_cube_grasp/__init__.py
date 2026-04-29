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

from panda_cube_grasp.envs import PandaCubeGraspEnv, PandaCubeGraspIKEnv

__all__ = ["PandaCubeGraspEnv", "PandaCubeGraspIKEnv"]
