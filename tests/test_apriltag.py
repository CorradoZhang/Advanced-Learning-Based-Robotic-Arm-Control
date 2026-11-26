"""Shared feature, pose and recorder contracts without hardware access."""
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
cv2=pytest.importorskip("cv2")
from vision.apriltag.detector import TagObservations, PREFIX, FEATURES
from vision.apriltag.features import apriltag_features,fit_normalization,normalize_apriltag
from vision.apriltag.recording import check_metadata,install


def calibration():
    return dict(camera_matrix=[[500,0,320],[0,500,240],[0,0,1]],distortion_coefficients=[0]*5,image_size=[640,480])


def test_rotation_columns_and_invalid_nan_masking():
    f=apriltag_features(np.array([[.1,.2,.3],[np.nan]*3]),np.array([[0,0,np.pi/2],[np.nan]*3]),[[1],[0]],[[.2],[np.nan]])
    np.testing.assert_allclose(f[0,3:9],[0,1,0,-1,0,0],atol=1e-6)
    np.testing.assert_array_equal(f[1],0)
    stats=fit_normalization(f,np.array([True,False]))
    np.testing.assert_array_equal(normalize_apriltag(f,stats)[1],0)
    with pytest.raises(ValueError): fit_normalization(f,np.array([False,True]))
    with pytest.raises(ValueError): apriltag_features([[0,0,0]],[[0,0,0]],[[.5]],[[0]])


def test_detector_pose_overlay_and_no_stale_detection():
    detector=TagObservations(calibration())
    marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9),0,120)
    gray=np.full((480,640),255,np.uint8)
    gray[180:300,260:380]=marker
    rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    raw=rgb.copy()
    obs=detector.extract(rgb)
    assert obs[PREFIX+'valid'][0]==1
    assert .08<obs[PREFIX+'position'][2]<.1
    preview=detector.annotate(rgb,True,obs)
    assert not np.array_equal(preview,rgb)
    np.testing.assert_array_equal(raw,rgb)
    blank=np.full_like(rgb,255)
    missing=detector.extract(blank)
    assert all(not v.any() for v in missing.values())
    np.testing.assert_array_equal(detector.annotate(blank,False,missing),blank)
    with pytest.raises(ValueError): detector.extract(rgb[:100])


def test_metadata_rejects_calibration_change(tmp_path):
    detector=TagObservations(calibration())
    root=tmp_path/'dataset'
    sidecar=check_metadata(root,detector)
    assert sidecar.exists()
    check_metadata(root,detector)
    with pytest.raises(ValueError): check_metadata(root,TagObservations(calibration(),tag_size=.03))


def test_recorder_uses_the_exact_raw_image_and_keeps_joint_schema(monkeypatch):
    image=np.zeros((480,640,3),np.uint8)
    calls=[]
    detector=SimpleNamespace(camera_key='wrist',extract=lambda rgb:(calls.append(rgb) or {key:np.zeros(spec['shape'],np.float32) for key,spec in FEATURES.items()}))
    recorder=SimpleNamespace(combine_feature_dicts=lambda: {'observation.images.wrist':{'shape':(480,640,3)}},
        build_dataset_frame=lambda features,values,prefix:{'joint_state':values.get('joints')})
    monkeypatch.setitem(sys.modules,'lerobot.scripts',SimpleNamespace(lerobot_record=recorder))
    install(detector)
    features=recorder.combine_feature_dicts()
    joints=np.arange(6)
    result=recorder.build_dataset_frame(features,{'wrist':image,'joints':joints},'observation')
    assert calls[0] is image and result['joint_state'] is joints
    assert all(key in result for key in FEATURES)
    recorder.build_dataset_frame(features,{'joints':joints},'action')
    assert len(calls)==1
    with pytest.raises(RuntimeError): install(detector)
