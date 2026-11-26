"""LeRobot recording entry point with bounded recovery for occasional stale camera frames."""

import logging
import time
import os
import sys

from pathlib import Path
import shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import Cv2Backends
from lerobot.scripts.lerobot_record import main


original_read_latest = OpenCVCamera.read_latest
original_connect = OpenCVCamera.connect
logger = logging.getLogger(__name__)


def connect_with_cleanup(self, *args, **kwargs):
    try:
        return original_connect(self, *args, **kwargs)
    except BaseException:
        # The pinned fork predates upstream connect-failure cleanup.
        try:
            if self.is_connected or self.thread is not None:
                self.disconnect()
        except Exception:
            logger.exception("Camera cleanup after failed connection was incomplete")
        raise


def check_camera(duration_s=30):
    """Use the official camera API before connecting either arm."""
    camera = OpenCVCamera(OpenCVCameraConfig(
        index_or_path=int(os.environ.get("CAMERA_INDEX", "0")),
        width=640, height=480, fps=int(os.environ.get("RECORD_FPS", "30")), warmup_s=2,
        backend=Cv2Backends.AVFOUNDATION,
    ))
    try:
        camera.connect()
        start = time.perf_counter()
        frames = 0
        max_age_ms = 0
        while time.perf_counter() - start < duration_s:
            camera.async_read(timeout_ms=1000)
            camera.read_latest(max_age_ms=500)
            with camera.frame_lock:
                max_age_ms = max(max_age_ms, (time.perf_counter() - camera.latest_timestamp) * 1000)
            frames += 1
        print(f"Camera check passed: {frames} frames in {duration_s} seconds; maximum frame age {max_age_ms:.1f} ms.", flush=True)
    finally:
        if camera.is_connected or camera.thread is not None:
            camera.disconnect()


def read_latest_with_recovery(self, max_age_ms=500):
    try:
        return original_read_latest(self, max_age_ms=max_age_ms)
    except TimeoutError as stale_error:
        logger.warning("%s frame is stale; waiting up to 500 ms for a fresh frame.", self)
        deadline = time.perf_counter() + 0.5
        while True:
            remaining_ms = (deadline - time.perf_counter()) * 1000
            if remaining_ms <= 0:
                raise TimeoutError(f"{self} did not deliver a fresh frame after an additional 500 ms.") from stale_error
            try:
                # async_read consumes the event, including any unconsumed event from an older frame.
                self.async_read(timeout_ms=remaining_ms)
                # Recheck frame age; recovery requires a frame no older than 100 ms to avoid saving stale images.
                return original_read_latest(self, max_age_ms=min(max_age_ms, 100))
            except TimeoutError:
                continue


if __name__ == "__main__":
    if "--check-camera" in sys.argv:
        check_camera(duration_s=int(os.environ.get("CAMERA_CHECK_SECONDS", "30")))
        sys.exit(0)
    OpenCVCamera.read_latest = read_latest_with_recovery
    OpenCVCamera.connect = connect_with_cleanup
    dataset_root = next((arg.split('=',1)[1] for arg in sys.argv if arg.startswith('--dataset.root=')),None)
    if dataset_root is None and '--dataset.root' in sys.argv:
        dataset_root = sys.argv[sys.argv.index('--dataset.root')+1]
    enabled = os.environ.get('APRILTAG_ENABLED','0') == '1'
    if enabled:
        from vision.apriltag.detector import TagObservations
        from vision.apriltag.recording import check_metadata, install
        if not dataset_root or not os.environ.get('APRILTAG_INTRINSICS'):
            raise ValueError('AprilTag recording needs --dataset.root and APRILTAG_INTRINSICS')
        detector = TagObservations(os.environ['APRILTAG_INTRINSICS'],
            tag_size=float(os.environ.get('APRILTAG_SIZE_M','.0215')),
            tag_id=int(os.environ.get('APRILTAG_ID','0')))
        check_metadata(dataset_root,detector)
        install(detector)
    elif dataset_root and (Path(str(dataset_root)+'.apriltag.json').exists() or
                            (Path(dataset_root)/'meta/apriltag.json').exists()):
        raise ValueError('Cannot resume an AprilTag dataset in RGB-only mode')
    try:
        main()
    finally:
        if enabled and dataset_root:
            sidecar = Path(str(dataset_root)+'.apriltag.json')
            destination = Path(dataset_root)/'meta'
            if sidecar.exists() and destination.is_dir():
                shutil.copy2(sidecar,destination/'apriltag.json')
