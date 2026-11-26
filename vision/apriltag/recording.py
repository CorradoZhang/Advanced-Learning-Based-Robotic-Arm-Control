"""LeRobot recorder adapter for the documented hardware fork."""
import json
from pathlib import Path
from .detector import FEATURES

def check_metadata(root, extractor):
    root = Path(root)
    path = root.parent / (root.name+'.apriltag.json')
    # JSON round-trip makes tuple feature shapes comparable across resumptions.
    metadata = json.loads(json.dumps(extractor.metadata()))
    existing = path if path.exists() else root/'meta/apriltag.json'
    if existing.exists():
        if json.loads(existing.read_text()) != metadata:
            raise ValueError('AprilTag configuration/calibration changed; start a new dataset')
    elif root.exists():
        raise ValueError('Existing dataset has no AprilTag schema; start a new dataset')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))
    return path


def install(extractor):
    from lerobot.scripts import lerobot_record as recorder
    if getattr(recorder, '_local_apriltag_installed', False):
        raise RuntimeError('AprilTag recorder already installed')
    original_combine = recorder.combine_feature_dicts
    original_build = recorder.build_dataset_frame

    def combine(*args, **kwargs):
        features = original_combine(*args, **kwargs)
        if 'observation.images.'+extractor.camera_key not in features:
            raise ValueError('AprilTag recording requires the configured wrist image feature')
        return {**features, **FEATURES}

    def build(features, values, prefix):
        # Keep the existing six-dimensional joint observation and action unchanged.
        base = {key: spec for key, spec in features.items() if key not in FEATURES}
        frame = original_build(base, values, prefix)
        if prefix == 'observation' and all(key in features for key in FEATURES):
            # The exact same RGB array is handed to the video writer in this frame.
            frame.update(extractor.extract(values[extractor.camera_key]))
        return frame

    recorder.combine_feature_dicts = combine
    recorder.build_dataset_frame = build
    recorder._local_apriltag_installed = True
