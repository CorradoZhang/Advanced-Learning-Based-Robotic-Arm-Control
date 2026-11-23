# Third-party notices

## ReinFlow

- Source: https://github.com/ReinFlow/ReinFlow
- Reference revision: `e722e151bed767f3ffef47527cf697f2358af55d`
- Paper: https://arxiv.org/abs/2505.22094
- License: MIT; retained in `il/flow_matching/LICENSE-ReinFlow` and source headers.
- Included/adapted components: `model.py`, `network.py`, `vit.py`, `mlp.py`,
  `modules.py`, `embedding.py`, `dataset.py`, `scheduler.py` under `il/flow_matching/`.
- Local adaptations: package-relative imports, compact model reconstruction,
  episode-bounded NPZ loading, batched intermediate sample storage, CPU-compatible
  attention backend selection, and the offline trainer/CLI integration.

This project uses the imitation-learning/pretraining part of the method. It does
not claim to reproduce the paper's online-RL experiments or benchmark results.

## Upstream helpers acknowledged by ReinFlow

Original notices are retained in individual files:

- ViT/spatial image modules: IBRL / Intelligent Robot Motion Lab,
  https://github.com/hengyuan-hu/ibrl
- Residual MLP: D3IL / ALRhub,
  https://github.com/ALRhub/d3il
- Cosine warmup/restart scheduler: Naoki Katsura,
  https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
- Sequence-window design: ReinFlow's loader, whose upstream references include
  https://github.com/jannerm/diffuser

## LeRobot and Hiwonder

- LeRobot: https://github.com/huggingface/lerobot
- Hiwonder fork: https://github.com/Hiwonder/hiwonder-SoArm-101
- Tested fork revision: `a24998f7ba3c77ea445b48c92ad15c14a50e492a`
- External dependency license: Apache-2.0; consult the installed distribution.
- The LeRobot/Hiwonder SDK source tree is not redistributed here.

The hardware adapter calls LeRobot's camera, calibration, motor and recording
interfaces. Hiwonder-specific support is for HX-30HM. Selecting STS3215 switches
to the fork's Feetech path; it is not a claim that every SO101 servo variant has
been tested or is mechanically/electrically interchangeable.
