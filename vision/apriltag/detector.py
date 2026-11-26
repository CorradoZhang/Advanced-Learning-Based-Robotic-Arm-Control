"""Same-image AprilTag auxiliary observations for the local LeRobot recorder."""
import json
from pathlib import Path
import cv2
import numpy as np
from .pose import estimate_pose

PREFIX = 'observation.apriltag.'
FEATURES = {
    PREFIX+'position': {'dtype': 'float32', 'shape': (3,), 'names': ['x_m', 'y_m', 'z_m']},
    PREFIX+'rotation': {'dtype': 'float32', 'shape': (3,), 'names': ['rx_rad', 'ry_rad', 'rz_rad']},
    PREFIX+'valid': {'dtype': 'float32', 'shape': (1,), 'names': ['valid']},
    PREFIX+'reprojection_error': {'dtype': 'float32', 'shape': (1,), 'names': ['error_px']},
}


class TagObservations:
    def __init__(self, intrinsics, tag_size=.0215, tag_id=0, camera_key='wrist'):
        self.calibration = intrinsics if isinstance(intrinsics, dict) else json.loads(Path(intrinsics).read_text())
        self.matrix = np.array(self.calibration['camera_matrix'], np.float64)
        self.distortion = np.array(self.calibration['distortion_coefficients'], np.float64)
        if self.matrix.shape != (3, 3) or not np.isfinite(self.matrix).all() or not np.isfinite(self.distortion).all():
            raise ValueError('Invalid camera calibration')
        if not np.isfinite(tag_size) or tag_size <= 0 or not 0 <= tag_id < 35:
            raise ValueError('Invalid tag size or tag25h9 ID')
        size = self.calibration.get('image_size')
        if not isinstance(size, list) or len(size) != 2 or any(not isinstance(v, int) or v <= 0 for v in size):
            raise ValueError('Calibration image_size must be [width, height]')
        if self.matrix[0, 0] <= 0 or self.matrix[1, 1] <= 0:
            raise ValueError('Focal lengths must be positive')
        if self.calibration.get('model','pinhole') != 'pinhole':
            raise ValueError('Only pinhole intrinsics are supported')
        self.tag_size, self.tag_id, self.camera_key = tag_size, tag_id, camera_key
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9), params)
        self.preview_detections = []

    def extract(self, rgb):
        self.preview_detections = []
        if rgb.dtype != np.uint8 or rgb.shape != (self.calibration['image_size'][1], self.calibration['image_size'][0], 3):
            raise ValueError('Wrist RGB resolution/dtype does not match calibration')
        output = {key: np.zeros(spec['shape'], np.float32) for key, spec in FEATURES.items()}
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return output
        self.preview_detections = [(int(id_), pts.reshape(4, 2).copy())
                                   for pts, id_ in zip(corners, ids.flatten())]
        matches = [pts for pts, id_ in zip(corners, ids.flatten()) if int(id_) == self.tag_id]
        # Duplicate IDs are ambiguous; never choose an arbitrary object.
        if len(matches) != 1:
            return output
        pose = estimate_pose(matches[0], self.matrix, self.distortion, self.tag_size)
        if pose is not None:
            output[PREFIX+'position'][:] = pose['xyz_m']
            output[PREFIX+'rotation'][:] = pose['rvec']
            output[PREFIX+'valid'][0] = 1
            output[PREFIX+'reprojection_error'][0] = pose['reprojection_error_px']
        return output

    def annotate(self, rgb, valid, observation=None):
        from .visualization import annotate
        return annotate(self, rgb, valid, observation)

    def metadata(self):
        return dict(schema_version=1, family='tag25h9', tag_id=self.tag_id,
            tag_size_m=self.tag_size, camera_key=self.camera_key,
            coordinates='OpenCV camera frame: X right, Y down, Z forward; tag center',
            rotation='Rodrigues axis-angle vector, radians; tag frame expressed in camera frame',
            invalid='valid=0: all auxiliary values zero; no previous pose reused',
            estimator='IPPE_SQUARE; positive depth; minimum reprojection RMS; reject >3 px',
            calibration=self.calibration, features=FEATURES,
            opencv_version=cv2.__version__)

