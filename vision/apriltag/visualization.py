"""Overlays on copies; raw RGB is never modified."""
import cv2
import numpy as np
PREFIX = "observation.apriltag."

def annotate(detector, rgb, valid, observation=None):
    """Draw only on a preview copy, using detections from the latest extract."""
    display = rgb.copy()
    for id_, corners in detector.preview_detections:
        target = id_ == detector.tag_id
        color = (0, 255, 0) if target and valid else ((255, 180, 0) if target else (180, 180, 180))
        xy = np.rint(corners).astype(np.int32)
        cv2.polylines(display, [xy], True, color, 2, cv2.LINE_AA)
        for point in xy:
            cv2.circle(display, tuple(point), 3, color, -1)
        x = max(0, min(int(xy[:, 0].min()), rgb.shape[1]-170))
        y = max(50, min(int(xy[:, 1].min())-10, rgb.shape[0]-10))
        label = f'ID {id_}' + (' (pose invalid)' if target and not valid else '')
        cv2.putText(display, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    if valid and observation is not None:
        # The preview is RGB; swap R/B around OpenCV drawing to retain conventional axis colors.
        bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        cv2.drawFrameAxes(bgr, detector.matrix, detector.distortion,
            observation[PREFIX+'rotation'].astype(np.float64),
            observation[PREFIX+'position'].astype(np.float64), detector.tag_size*.75)
        display = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return display

