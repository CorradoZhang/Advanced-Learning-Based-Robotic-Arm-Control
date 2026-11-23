# Advanced Learning-Based Robotic Arm Control

A research project for learning-based robotic arm control, bringing together
robot hardware, data collection, learning algorithms, evaluation and real-world
execution. SO101 is the current development platform.

The project is organized around **imitation learning (IL)** and **reinforcement
learning (RL)** as learning categories. Individual methods live within their
category. Vision-language-action (VLA) policies are a future extension whose
training and deployment workflow will be placed according to the method used.

**Current release:** the implemented learning method is Visual Flow Matching
under IL, with demonstration collection, training, offline prediction and SO101
deployment. RL and VLA are planned directions and have no implementation yet.

[中文使用指南](docs/README.zh-CN.md) · [IL / Flow Matching tutorial](docs/il/flow_matching.md) ·
[Hardware setup](docs/hardware.md) · [Data and training](docs/training.md)

## Project scope

The project connects the main stages of robotic learning research:

- **Hardware and interaction:** configuration, calibration, teleoperation and camera access.
- **Data:** demonstration recording, dataset preparation and training/validation splits.
- **Learning:** method implementations, experiment configuration and training entry points.
- **Evaluation:** offline policy predictions and experiment artifacts.
- **Deployment:** camera observations, policy inference and physical robot control.

These capabilities currently support the SO101 Flow Matching workflow. Shared
components can be extended as additional methods are implemented; data formats
and deployment interfaces are currently tailored to that workflow.

## Architecture

Learning methods and their execution entry points are separated. Model and
trainer code live under the learning category; runners expose preparation,
training and prediction commands. Configuration, hardware tools and deployment
have their own directories.

```text
il/
└── flow_matching/             # Implemented IL method: model, dataset and trainer
runner/
└── il/flow_matching/          # Preparation, training and prediction entry points
configs/
├── il/flow_matching.yaml      # Method configuration
└── hardware.example.yaml     # Hardware configuration template
scripts/                      # Calibration, teleoperation, recording, camera tools
deployment/                   # Current SO101 policy execution
tests/                        # Hardware-free workflow checks
docs/                         # Method tutorials and hardware documentation
```

Only implemented modules appear in the source tree. A future RL implementation
can add its own category and runners when there is working code to include.

## Implementation status

| Area | Status |
|---|---|
| SO101 hardware tools | Calibration, teleoperation, demonstration recording and camera preview |
| Imitation learning | Visual Flow Matching: preparation, training, validation and prediction |
| Physical deployment | SO101 inference, preview, command limits and fault checks |
| Reinforcement learning | Planned; no trainer or environment integration yet |
| VLA integration | Planned; no pretrained model integration yet |

Hiwonder HX-30HM leader/follower arms and a wrist RGB camera have been used during
development. STS3215 can be selected through the hardware adapter but has not
been physically validated in this project. The current policy uses five joint
angles and a calibrated gripper coordinate; the interfaces do not yet support
arbitrary robot platforms.

## Getting started

Install the project in a Python 3.10+ environment with a suitable PyTorch build:

```bash
git clone https://github.com/CorradoZhang/Advanced-Learning-Based-Robotic-Arm-Control.git
cd Advanced-Learning-Based-Robotic-Arm-Control
python -m pip install -e '.[prepare,test]'
```

For physical robot access, also install the pinned hardware dependency from
`requirements-hardware.txt`. Training and hardware environments can be separate;
development used Linux/CUDA training and macOS/CPU inference.

Choose the documentation for your task:

| Task | Guide |
|---|---|
| Configure hardware, calibrate and collect demonstrations | [Hardware guide](docs/hardware.md) |
| Run the implemented learning workflow | [IL / Visual Flow Matching](docs/il/flow_matching.md) |
| Understand datasets, normalization and training outputs | [Data and training](docs/training.md) |
| Follow collection, training and deployment in Chinese | [中文使用指南](docs/README.zh-CN.md) |

## Research and extension

Additional IL methods can be added alongside `il/flow_matching/`. RL methods can
be introduced in their own learning category, with their required environments,
training entry points and evaluation. A VLA integration will need its own model,
observation/action adaptation and instructions for training or inference.

Each implemented method should document its inputs, action representation,
training procedure, evaluation and deployment requirements. Shared interfaces
should grow from those concrete requirements.

## Verification and limitations

```bash
python -m pytest
```

Tests cover synthetic data conversion, CPU training, checkpoint resume,
prediction and simulated controller guards. They do not actuate hardware.
Recordings, model weights and device calibration files are not bundled.

The current SO101 workflow remains experimental. Camera stalls, gripper overload
and sensitivity to initial pose and viewpoint require further work. Validation
loss is not a physical task success rate; software limits and fault checks do
not replace a hardware emergency stop.

## License and acknowledgments

Project integration code is MIT licensed. The current Flow Matching
implementation adapts code from [ReinFlow](https://github.com/ReinFlow/ReinFlow)
and retains the original attribution and license. See
[third-party notices](THIRD_PARTY_NOTICES.md) for provenance.
Earlier Panda/MuJoCo prototype code remains in Git history.
