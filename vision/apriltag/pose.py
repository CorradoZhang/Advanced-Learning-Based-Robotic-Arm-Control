"""Metric square-tag pose in OpenCV camera coordinates: right, down, forward."""
import cv2
import numpy as np


def estimate_pose(corners, matrix, distortion, tag_size_m):
    half = tag_size_m / 2
    # ArUco corner order: top-left, top-right, bottom-right, bottom-left.
    obj = np.array([[-half, half, 0], [half, half, 0],
                    [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
    image = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    ok, rotations, translations, _ = cv2.solvePnPGeneric(
        obj, image, matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    candidates = []
    for r, t in zip(rotations, translations):
        rotation, _ = cv2.Rodrigues(r)
        if not np.isfinite(t).all() or np.any((obj @ rotation.T + t.reshape(1, 3))[:, 2] <= 0):
            continue
        projected, _ = cv2.projectPoints(obj, r, t, matrix, distortion)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2)-image)**2, axis=1))))
        candidates.append((error, r, t))
    if not candidates:
        return None
    error, r, t = min(candidates, key=lambda item: item[0])
    if not np.isfinite(error) or error > 3:
        return None
    return dict(rvec=r.reshape(3).tolist(), xyz_m=t.reshape(3).tolist(),
                distance_m=float(np.linalg.norm(t)), reprojection_error_px=error)
