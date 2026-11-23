# Hardware setup and troubleshooting

## Configuration

Use the optional pinned Hiwonder LeRobot fork in a dedicated environment.
The generic upstream SO101 implementation does not automatically provide the
same HX-30HM driver. Keep the environment and calibration used for recording.

Set the actual leader/follower serial ports in `hardware.local.yaml`. Port names
can change after reconnecting a device. Persistently use the same robot IDs for
calibration, recording and deployment. Never reuse another arm's calibration file.
For deployment, pass `--port`, `--robot-id`, `--camera-index` and `--motor-model`.

Camera indexes are backend-specific and can change. Two USB cameras may have the
same product name. On macOS, `system_profiler SPCameraDataType` lists device names
and unique IDs without starting a video stream. A separate FFplay preview is
available, with an explicit device selector:

```bash
# macOS only; install FFmpeg separately (e.g. brew install ffmpeg).
python -m scripts.camera_preview --camera YOUR_AVFOUNDATION_DEVICE_ID
```

FFmpeg indexes need not match OpenCV indexes. Close the preview before recording
or deployment. Camera discovery utilities can also enumerate phone/Continuity cameras.

## Units and limits

With the supplied collection settings, the first five joint coordinates are in
degrees. The gripper coordinate is calibrated 0–100, not degrees or a direct
torque command. A model's [-1,1] output is inverted using its own training
statistics, not scaled blindly to a full physical range.

Runtime checks include:

- Per-motor acknowledged reads before execution, plus checks every 0.25 seconds.
- Current-pose distribution checks and a maximum observation age of 500 ms.
- Joint target slew limits (30 degrees/s; gripper 30 scale points/s by default).
- A tracking-distance check relative to measured joints.
- Encoder goals checked after quantization against hardware/calibration bounds.
- A one-tick inward endpoint margin; excessive correction stops execution.

Model target bounds and servo hardware limits are different. A slightly
out-of-training-range pose can approach a predicted target gradually while
hardware bounds still constrain every issued command. The script never homes
or resets the arm automatically. No force or collision safety guarantee is made.

## Troubleshooting

### Camera timeout or `read failed`

An explicit `TimeoutError` or failed frame read is a capture failure. A plain
`OpenCVCamera disconnected` at the end of `--duration` is normal cleanup.
Avoid concurrent camera users. Compare stationary capture with capture during
arm movement; secure the wrist cable with enough slack. Test direct USB versus
a dock and another cable. USB, driver and backend causes must be distinguished
by tests; increasing a timeout does not establish or fix the root cause.

The recording wrapper waits a bounded extra 500 ms for a fresh frame. It raises
if capture does not recover, rather than saving stale images indefinitely.
Already committed episodes can be resumed after the camera is stable again.

### `Missing motor ID 6` / `Overload error`

The driver's ping can classify a servo returning fault bits as missing. Run
`--diagnose` to distinguish a servo status fault from loss of communication.
A gripper overload may remain reported even when torque is off. Support the
arm, remove the load and inspect mechanical closure/calibration before power
cycling. Do not skip the fault or repeatedly run the policy into sustained contact.

### Arm still holds after stopping

Support the arm and use `--release`. The command tries each motor and reads back
the torque state; `torque OFF verified` is the confirmation. If confirmation
fails, support the arm and disconnect servo power. USB disconnection alone may
not remove external servo power. A hard kill, native crash or USB failure may
prevent cleanup, so keep a physical power cutoff accessible.

### Angle errors at startup

Do not expand stored servo limits to hide a fault. The controller reads raw
encoder positions for its initial holding targets and checks goals after
conversion. This avoids the float32 endpoint roundtrip where a valid tick 905
can become the invalid integer 904. Large starting-pose corrections are rejected.

### Software scope

HX-30HM behavior has been exercised during development. STS3215 is selectable
through the fork's existing Feetech path but requires new hardware-specific
calibration and testing. Different voltage/gear-ratio variants are not assumed
interchangeable. Camera, overload and native-crash regressions remain experimental
and should be reported with logs, environment versions and hardware details.
