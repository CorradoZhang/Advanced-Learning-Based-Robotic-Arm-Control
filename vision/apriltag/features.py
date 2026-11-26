"""Shared training/deployment features: position, rotation columns, validity and error."""
import numpy as np

FEATURE_ORDER = ['x_m', 'y_m', 'z_m', 'r00', 'r10', 'r20',
                 'r01', 'r11', 'r21', 'valid', 'reprojection_error_px']


def apriltag_features(position, rotation, valid, reprojection_error):
    valid = np.asarray(valid, dtype=np.float32).reshape(-1, 1)
    if not np.isin(valid, [0, 1]).all():
        raise ValueError('AprilTag valid must contain only 0 or 1')
    if np.asarray(position).shape != (len(valid),3) or np.asarray(rotation).shape != (len(valid),3) or np.asarray(reprojection_error).size != len(valid):
        raise ValueError('Expected position/rotation [N,3] and valid/error [N,1]')
    position = np.where(valid == 1, np.asarray(position, dtype=np.float32), 0.)
    rotation = np.where(valid == 1, np.asarray(rotation, dtype=np.float32), 0.)
    reprojection_error = np.where(valid == 1, np.asarray(reprojection_error).reshape(-1, 1), 0.)
    if any(not np.isfinite(v).all() for v in (position, rotation, reprojection_error)):
        raise ValueError('Valid AprilTag observations must be finite')
    theta = np.linalg.norm(rotation, axis=-1)
    skew = np.zeros(rotation.shape[:-1] + (3, 3), dtype=np.float32)
    x, y, z = np.moveaxis(rotation, -1, 0)
    skew[..., 0, 1], skew[..., 0, 2] = -z, y
    skew[..., 1, 0], skew[..., 1, 2] = z, -x
    skew[..., 2, 0], skew[..., 2, 1] = -y, x
    matrix = (np.eye(3, dtype=np.float32)
              + np.sinc(theta / np.pi)[..., None, None] * skew
              + (0.5 * np.sinc(theta / (2 * np.pi)) ** 2)[..., None, None] * (skew @ skew))
    valid = np.asarray(valid, dtype=np.float32).reshape(-1, 1)
    pose = np.concatenate((position, matrix[..., :, 0], matrix[..., :, 1]), axis=-1)
    return np.concatenate((pose * valid, valid,
                           np.asarray(reprojection_error).reshape(-1, 1) * valid), axis=-1).astype(np.float32)


def normalize_apriltag(features, stats):
    normalized = (features - stats['apriltag_mean']) / stats['apriltag_std']
    normalized *= features[:, 9:10]
    normalized[:, 9] = features[:, 9]
    return normalized.astype(np.float32)


def observation_features(observation, stats):
    prefix = 'observation.apriltag.'
    features = apriltag_features(*(observation[prefix+key][None] for key in
        ('position', 'rotation', 'valid', 'reprojection_error')))
    return normalize_apriltag(features, stats)[0]


def fit_normalization(features, train_mask):
    """Fit valid training frames only; validation frames never influence statistics."""
    values = features[np.asarray(train_mask) & (features[:, 9] == 1)]
    if not len(values):
        raise ValueError('Need valid training detections to fit AprilTag statistics')
    mean, std = values.mean(0), values.std(0)
    mean[9], std[9] = 0, 1
    std = np.where(std > 1e-6, std, 1).astype(np.float32)
    return dict(apriltag_mean=mean, apriltag_std=std)


def validate_metadata(metadata):
    """Check the portable contract without importing OpenCV on a training server."""
    if metadata.get('family') != 'tag25h9' or metadata.get('camera_key') != 'wrist':
        raise ValueError('Expected tag25h9 observations from the wrist camera')
    if not isinstance(metadata.get('tag_id'),int) or not 0 <= metadata['tag_id'] < 35:
        raise ValueError('Invalid tag25h9 ID')
    size = metadata.get('tag_size_m',0)
    if not np.isfinite(size) or size <= 0:
        raise ValueError('Tag size must be positive meters')
    calibration=metadata['calibration']
    matrix=np.asarray(calibration['camera_matrix'])
    if matrix.shape != (3,3) or not np.isfinite(matrix).all() or min(matrix[0,0],matrix[1,1]) <= 0:
        raise ValueError('Invalid camera matrix')
    if calibration.get('model','pinhole') != 'pinhole':
        raise ValueError('Only pinhole intrinsics are supported')
    if calibration['image_size'] != [640,480]:
        raise ValueError('The current recording/deployment workflow requires 640x480 intrinsics')
    distortion=np.asarray(calibration['distortion_coefficients'])
    if distortion.size not in (4,5,8,12,14) or not np.isfinite(distortion).all():
        raise ValueError('Invalid pinhole distortion coefficients')
