from panda_cube_grasp.envs.panda_cube_grasp_env import (
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
