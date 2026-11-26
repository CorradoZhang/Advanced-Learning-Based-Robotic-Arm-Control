# Optional AprilTag visual assistance

The RGB baseline conditions Flow Matching on six joint positions. The assisted
variant appends 11 AprilTag features, giving a 17-dimensional condition. Both
predict 16 actions, each containing five joint angles in degrees and one gripper
position in calibrated 0–100 units. Smaller gripper values mean more closed.

This module estimates tag pose in **camera coordinates**, not gripper or base
coordinates. It does not implement hand-eye calibration or force sensing.
The external camera used for presentation videos is not a second policy input.

## Install

From the repository root, in your training environment:

```bash
pip install -e '.[prepare,vision,test]'
```

Hardware recording additionally needs the LeRobot fork pinned in
`requirements-hardware.txt`. The recorder adapter targets that fork's
`combine_feature_dicts` and `build_dataset_frame` API. Training and offline
prediction do not import serial or camera drivers. The features module only
requires NumPy; OpenCV is needed for detection and calibration.
If your hardware environment already supplies OpenCV with `cv2.aruco`, use that
installation instead of installing multiple OpenCV wheel variants together.

## 1. Calibrate the wrist camera

Print a flat checkerboard, measure one square, and use its **inner corner count**.
The example is 9×6 inner corners with 20 mm squares. Use diverse positions,
orientations and distances, covering the image. Twenty or more views are
recommended. Space captures, Enter solves/saves, Escape exits.

```bash
python -m scripts.calibrate_camera --camera-index 0 \
  --columns 9 --rows 6 --square-mm 20 \
  --out calibration/wrist/intrinsics.json
```

On macOS, optionally add `--backend 1200`. Choose the correct OpenCV camera index;
FFmpeg and OpenCV indices are independent. Intrinsics are specific to camera,
resolution and focus. Use the same 640×480 resolution and fixed camera mounting
for collection and deployment. This is a pinhole model, not a fisheye model.
No user-specific intrinsics are shipped as a universal calibration.

## 2. Verify tag detection

```bash
python -m scripts.preview_apriltag --camera-index 0 \
  --intrinsics calibration/wrist/intrinsics.json \
  --tag-size-m 0.0215 --tag-id 0
```

The supported dictionary is `tag25h9` (IDs 0–34). The condition tracks one
configured ID; other detected IDs can appear in the preview. Measure the square
outer black border, excluding the surrounding white margin. `0.0215` means
21.5 mm; change it to your measured size. Incorrect size scales estimated distance.
Green indicates a valid target pose. Duplicate target IDs, missing tags, negative
depth, or reprojection RMS above 3 px produce `valid=0`.

## 3. Record raw RGB and auxiliary observations

Use your hardware YAML as described in [hardware.md](hardware.md). This command
starts real leader/follower teleoperation and dataset recording:

```bash
APRILTAG_ENABLED=1 \
APRILTAG_INTRINSICS=calibration/wrist/intrinsics.json \
APRILTAG_SIZE_M=0.0215 APRILTAG_ID=0 \
python -m scripts.hardware --config configs/hardware.local.yaml record \
  --episodes 30 --name pick_place_tag --task 'Pick and place the object'
```

The feature extractor uses the **same RGB array** saved for that observation.
Original images are not annotated. The six-dimensional state/action schema stays
unchanged; extra dataset columns are:

| Dataset key | Shape | Units |
|---|---|---|
| `observation.apriltag.position` | 3 | meters, tag center in camera frame |
| `observation.apriltag.rotation` | 3 | Rodrigues rotation vector, radians |
| `observation.apriltag.valid` | 1 | binary 0/1 |
| `observation.apriltag.reprojection_error` | 1 | pixels, corner RMS |

Camera axes are X right, Y down, Z forward. The dataset sidecar
`DATASET.apriltag.json` is also copied to `DATASET/meta/apriltag.json` on exit.
It includes intrinsics, resolution, tag size, ID and family. Copy that metadata
with the dataset. Resume with the same environment variables and
`record --resume datasets/pick_place_tag --episodes 10 --task 'Pick and place the object'`.
Changing tag/calibration metadata is rejected, as is resuming a tagged dataset
with assistance disabled. RGB recording remains the default (`APRILTAG_ENABLED=0`).

## 4. Prepare and train

```bash
python -m runner.il.flow_matching.prepare --dataset datasets/pick_place_tag \
  --out data/pick_place_tag --val-episodes 3 --apriltag
python -m runner.il.flow_matching.train --data data/pick_place_tag \
  --epochs 200 --batch-size 64 --device cuda:0
```

The training entry point selects `flow_matching_apriltag.yaml` for 17-D data and
`flow_matching_rgb.yaml` for 6-D data. You may pass `--config` explicitly;
config/data dimension mismatches are rejected. The original `flow_matching.yaml`
is retained for existing RGB commands.

The 11 appended features, in order, are:

```text
x_m y_m z_m r00 r10 r20 r01 r11 r21 valid reprojection_error_px
```

The six rotation values are the first two **columns** of the tag-to-camera
rotation matrix, in column order. State/action min-max statistics use training
episodes only. Auxiliary z-score statistics use **valid training frames only**;
validity remains binary. Missing detections stay zero after normalization, with
no last-pose reuse. At least one valid training detection is required.
Normalization, feature schema and calibration travel with each training run.

For a fair RGB baseline, convert the **same raw dataset**, seed and validation
count into a separate output directory without `--apriltag`. RGB images still
contain the printed marker: this comparison measures the value of explicit pose
features, not the effect of removing the marker from the scene. Report task
success over repeated trials and visibility/occlusion conditions; validation
loss alone is not a task success rate. No benchmark results are claimed here.

## 5. Predict and deploy

Offline, with no camera or robot:

```bash
python -m runner.il.flow_matching.predict --run result/il/flow_matching/YOUR_RUN \
  --data data/pick_place_tag/val.npz --device cpu
python -m deployment.deploy --run result/il/flow_matching/YOUR_RUN \
  --offline-data data/pick_place_tag/val.npz --no-preview
```

Live read-only preview (default, no motor writes):

```bash
python -m deployment.deploy --run result/il/flow_matching/YOUR_RUN \
  --port YOUR_FOLLOWER_PORT --robot-id YOUR_CALIBRATED_ID --camera-index 0
```

Only add `--execute` when ready for real motion. Tagged checkpoints automatically
load the saved detector configuration, annotate a preview copy and construct the
same 17-D input used during training. RGB checkpoints do not invoke the detector.
Checkpoints, configuration, statistics, manifest and physical camera setup must
match. Calibrating the camera does not repair low grip force or policy errors.
Changing the coordinate representation requires a consistent new training run.

## Source layout and checks

`vision/apriltag/detector.py` handles detection and metadata; `pose.py` handles
IPPE pose estimation; `features.py` defines the shared training/deployment
representation; `recording.py` adapts the LeRobot recorder; `visualization.py`
draws on copies. No personal calibration, demonstrations, weights or recorded
videos are required in Git.

```bash
python -m pytest
```

Tests use synthetic images, video/Parquet fixtures and simulated hardware. They
cover both RGB and assisted conversion, CPU training, checkpoint loading, offline
prediction, invalid observations, rotation order and recorder hooks. These are
software checks, not a real-robot performance evaluation.
