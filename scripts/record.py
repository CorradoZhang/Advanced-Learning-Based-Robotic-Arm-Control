"""LeRobot 录制入口，为偶发的旧相机帧增加有界的新帧等待。"""

import logging
import time
import os
import sys

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
        print(f"相机检查通过：{frames} 帧，持续 {duration_s} 秒，最大帧龄 {max_age_ms:.1f} ms。", flush=True)
    finally:
        if camera.is_connected or camera.thread is not None:
            camera.disconnect()


def read_latest_with_recovery(self, max_age_ms=500):
    try:
        return original_read_latest(self, max_age_ms=max_age_ms)
    except TimeoutError as stale_error:
        logger.warning("%s 画面过期，等待新帧（最多 500 ms）。", self)
        deadline = time.perf_counter() + 0.5
        while True:
            remaining_ms = (deadline - time.perf_counter()) * 1000
            if remaining_ms <= 0:
                raise TimeoutError(f"{self} 在额外等待 500 ms 后仍未恢复新画面。") from stale_error
            try:
                # async_read 会消费事件，包括之前未消费的旧帧事件。
                self.async_read(timeout_ms=remaining_ms)
                # 再检查帧龄，恢复时要求画面在 100 ms 内，避免保存旧帧。
                return original_read_latest(self, max_age_ms=min(max_age_ms, 100))
            except TimeoutError:
                continue


if __name__ == "__main__":
    if "--check-camera" in sys.argv:
        check_camera(duration_s=int(os.environ.get("CAMERA_CHECK_SECONDS", "30")))
        sys.exit(0)
    OpenCVCamera.read_latest = read_latest_with_recovery
    OpenCVCamera.connect = connect_with_cleanup
    main()
