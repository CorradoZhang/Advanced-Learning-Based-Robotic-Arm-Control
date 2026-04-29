# Third-Party Notices

This project includes third-party files and derived assets.

## MuJoCo Menagerie: Franka Emika Panda

Included path:
- `mujoco_menagerie/franka_emika_panda/`

Primary upstream source:
- Repository: `google-deepmind/mujoco_menagerie`
- URL: `https://github.com/google-deepmind/mujoco_menagerie`
- Model directory: `https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda`

License:
- Apache License 2.0
- The upstream license text is retained at:
  `mujoco_menagerie/franka_emika_panda/LICENSE`

Upstream provenance noted by the model authors:
- The `franka_emika_panda` model README states that this MJCF model is derived from the publicly available URDF description in `frankarobotics/franka_ros` / `franka_description`.
- `franka_ros`: `https://github.com/frankarobotics/franka_ros`
- `franka_description`: `https://github.com/frankarobotics/franka_description`

Local modifications in this repository:
- Added `mujoco_menagerie/franka_emika_panda/grasp_scene.xml`
- Added a MuJoCo grasp demo script in `test.py`
- Adjusted table, cube, and camera settings for the local grasp experiment

## Suggested Citation

If you use the MuJoCo Menagerie assets in academic work, cite:

```bibtex
@software{menagerie2022github,
  author = {Zakka, Kevin and Tassa, Yuval and {MuJoCo Menagerie Contributors}},
  title = {{MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo}},
  url = {http://github.com/google-deepmind/mujoco_menagerie},
  year = {2022},
}
```

Reference files retained locally:
- `mujoco_menagerie/CITATION.cff`
- `mujoco_menagerie/franka_emika_panda/README.md`
- `mujoco_menagerie/franka_emika_panda/LICENSE`
