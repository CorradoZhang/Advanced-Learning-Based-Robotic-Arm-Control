"""Controller guards tested with simulated hardware; never opens serial or camera."""
import importlib.util
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from deployment.deploy import JOINTS, limited_target, live, check_servos, close_bus, bounded_raw, raw_action, release, canonical_config, check_execution_health


HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None


class DeployTests(unittest.TestCase):
    def test_live_health_catches_gripper_overload(self):
        motors = {str(i):SimpleNamespace(id=i) for i in range(1,7)}
        def read(key,name):
            if name=='6':
                raise RuntimeError('Overload error')
            return 1
        with self.assertRaisesRegex(RuntimeError,'id=6.*Overload'):
            check_execution_health(SimpleNamespace(motors=motors,read=read))
        with self.assertRaisesRegex(RuntimeError,'disabled unexpectedly'):
            check_execution_health(SimpleNamespace(motors=motors,read=lambda *_:0))

    def test_initial_pose_outside_training_range_approaches_without_jump(self):
        stats = dict(action_min=np.full(6,-10.),action_max=np.full(6,10.))
        current = np.full(6,14.)
        previous = current.copy()
        for _ in range(4):
            result = limited_target(np.full(6,100.),current,previous,1/30,30,stats,30,30)
            self.assertTrue(np.all(np.abs(result-previous) <= 1.00001))
            self.assertTrue(np.all(result <= previous))
            previous = result
            current = result.copy()
        np.testing.assert_allclose(previous,10.)

    def test_compact_config_preserves_training_dimensions(self):
        compact = dict(model=dict(action_dim=6,obs_dim=6,horizon_steps=16,
            network=dict(backbone=dict(obs_shape=[3,96,96]))),
            train_dataset=dict(cond_steps=1,img_cond_steps=1))
        c = canonical_config(compact)
        self.assertEqual(c['action_dim'],6)
        self.assertEqual(c['cond_steps'],1)
        self.assertEqual(c['shape_meta']['obs']['rgb']['shape'],[3,96,96])
        self.assertNotIn('cond_steps',compact)
        legacy = dict(compact,cond_steps=2,img_cond_steps=1,action_dim=6,obs_dim=6,
            horizon_steps=16,shape_meta={'obs':{'rgb':{'shape':[3,96,96]}}})
        self.assertEqual(canonical_config(legacy)['cond_steps'],2)

    @unittest.skipUnless(HAS_LEROBOT,"optional LeRobot package not installed")
    def test_release_skips_handshake_and_attempts_every_servo(self):
        from lerobot.robots.so_follower import so_follower
        calls = []
        motors = {str(i):SimpleNamespace(id=i) for i in range(1,7)}
        def disable(name,**kwargs):
            calls.append(name)
            if name == '2':
                raise RuntimeError('fault')
        bus = SimpleNamespace(is_connected=True,motors=motors,
            connect=lambda **kw:calls.append(kw),disable_torque=disable,
            read=lambda *_:0,disconnect=lambda **kw:calls.append(kw))
        with patch.object(so_follower,'SOFollower',lambda *_:SimpleNamespace(bus=bus)):
            with self.assertLogs('so101_policy',level='ERROR'):
                with self.assertRaisesRegex(RuntimeError,'id=2'):
                    release(SimpleNamespace(port='fake',robot_id='test'))
        self.assertEqual(calls[0],dict(handshake=False))
        self.assertEqual(calls[1:7],list(motors))
        self.assertEqual(calls[-1],dict(disable_torque=False))

    def test_endpoint_quantization_cannot_write_outside_limits(self):
        motors = {j.removesuffix(".pos"):SimpleNamespace(id=i+1) for i,j in enumerate(JOINTS)}
        def unnormalize(values):
            result = {i:2000 for i in range(1,7)}
            result[2] = int(values[2]*4095/360+(905+3280)/2)
            result[3] = int(values[3]*4095/360+(766+2964)/2)
            return result
        bus = SimpleNamespace(motors=motors,_unnormalize=unnormalize)
        target = np.zeros(6,dtype=np.float32)
        target[1] = (905-(905+3280)/2)*360/4095
        target[2] = (2965-(766+2964)/2)*360/4095
        limits = {n:(1,4094) for n in motors}
        limits['shoulder_lift'] = (906,3279)
        limits['elbow_flex'] = (767,2963)
        self.assertEqual(unnormalize({i+1:float(v) for i,v in enumerate(target)})[2],904)
        raw = raw_action(bus,target,limits)
        self.assertEqual(raw['shoulder_lift'],906)
        self.assertEqual(raw['elbow_flex'],2963)
        with self.assertRaises(RuntimeError):
            bounded_raw({'shoulder_lift':800},limits)

    def test_fault_check_reads_all_motors_before_execution(self):
        calls = []
        motors = {str(i):SimpleNamespace(id=i) for i in range(1,7)}
        def read(key,name):
            calls.append(name)
            if name == "2":
                raise RuntimeError("Angle error")
            return 0
        with self.assertRaisesRegex(RuntimeError,"id=2"):
            check_servos(SimpleNamespace(motors=motors,read=read))
        self.assertEqual(calls,list(motors))

    def test_shutdown_continues_after_one_fault(self):
        calls = []
        motors = {str(i):SimpleNamespace(id=i) for i in range(1,7)}
        def disable(name,**kwargs):
            calls.append(name)
            if name == "2":
                raise RuntimeError("Angle error")
        bus = SimpleNamespace(is_connected=True,motors=motors,disable_torque=disable,
            disconnect=lambda **kw:calls.append(kw))
        with self.assertLogs("so101_policy",level="ERROR"):
            close_bus(bus,True)
        self.assertEqual(calls[:6],list(motors))
        self.assertEqual(calls[-1],dict(disable_torque=False))

    def test_slew_and_training_bounds(self):
        stats = dict(action_min=np.full(6,-10),action_max=np.full(6,10))
        v = limited_target(np.full(6,100),np.zeros(6),np.zeros(6),100,30,stats,30,15)
        np.testing.assert_allclose(v,[2,2,2,2,2,1])
        with self.assertRaises(ValueError):
            limited_target(np.full(6,np.nan),np.zeros(6),np.zeros(6),.03,30,stats,30,15)
        with self.assertRaises(RuntimeError):
            limited_target(np.zeros(6),np.full(6,20),np.zeros(6),.03,30,stats,30,15)

    def exercise_read_only(self, stale=False):
        from lerobot.robots.so_follower import so_follower
        from lerobot.cameras.opencv import camera_opencv
        calls = []
        class Bus:
            is_connected = False
            def connect(self):
                self.is_connected = True
            def sync_read(self, key):
                calls.append("read")
                return {j.removesuffix(".pos"):0.0 for j in JOINTS}
            def sync_write(self,*args):
                raise AssertionError("Read-only mode attempted a motor write")
            def disconnect(self,disable_torque):
                calls.append(("disconnect",disable_torque))
                self.is_connected = False
        class Robot:
            def __init__(self,*args):
                self.bus = Bus()
                self.calibration = True
                self.is_calibrated = True
            def configure(self):
                raise AssertionError("Read-only mode attempted torque configuration")
            def send_action(self,*args):
                raise AssertionError("Read-only mode attempted an action")
        class Camera:
            is_connected = False
            thread = None
            def __init__(self,*args):
                self.frame_lock = threading.Lock()
                self.latest_frame = np.zeros((4,4,3),dtype=np.uint8)
                self.latest_timestamp = time.perf_counter()-(2 if stale else 0)
            def connect(self):
                self.is_connected = True
            def read_latest(self,**kwargs):
                return self.latest_frame
            def disconnect(self):
                calls.append("camera_closed")
                self.is_connected = False
        policy = SimpleNamespace(fps=30,normalize_state=lambda _:np.zeros(6),
                                 predict=lambda *_:np.zeros((16,6)))
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(port="fake",robot_id="test",camera_index=0,
                execute=False,preview=False,duration=.01,log=Path(tmp)/"test.jsonl")
            with patch.object(so_follower,"SOFollower",Robot),patch.object(camera_opencv,"OpenCVCamera",Camera):
                if stale:
                    with self.assertRaises(TimeoutError):
                        live(args,policy)
                else:
                    live(args,policy)
        self.assertIn(("disconnect",False),calls)
        self.assertIn("camera_closed",calls)

    @unittest.skipUnless(HAS_LEROBOT,"optional LeRobot package not installed")
    def test_read_only_never_writes_or_changes_torque(self):
        self.exercise_read_only()

    @unittest.skipUnless(HAS_LEROBOT,"optional LeRobot package not installed")
    def test_expired_inference_stops_and_releases_resources(self):
        self.exercise_read_only(stale=True)


if __name__ == "__main__":
    unittest.main()
