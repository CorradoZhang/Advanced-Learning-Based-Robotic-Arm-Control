"""Local SO101 closed-loop inference. Default: read-only for 30 seconds; --execute enables motion."""
import argparse
import copy
import json
import logging
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
JOINTS = [f"{name}.pos" for name in (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
DATA = ROOT / "artifacts"
log = logging.getLogger("so101_policy")


def canonical_config(config):
    """Support the original Hydra layout and the newer compact training layout."""
    cfg = copy.deepcopy(config)
    dataset = cfg.get("train_dataset",{})
    model = cfg["model"]
    for name in ("cond_steps","img_cond_steps"):
        if name not in cfg:
            cfg[name] = dataset[name]
    for name in ("action_dim","obs_dim","horizon_steps"):
        if name not in cfg:
            cfg[name] = model[name]
    if "shape_meta" not in cfg:
        cfg["shape_meta"] = {"obs":{"rgb":{"shape":model["network"]["backbone"]["obs_shape"]}}}
    return cfg


class Policy:
    def __init__(self, checkpoint, config, stats, manifest, device="cpu", weights="model", steps=20):
        # Use official modules directly; no Hydra/simulator dependencies on the recording Mac.
        from il.flow_matching.vit import VitEncoder
        from il.flow_matching.network import VisionFlowMLP
        from il.flow_matching.model import ReFlow
        cfg = canonical_config(yaml.safe_load(Path(config).read_text()))
        meta = json.loads(Path(manifest).read_text())
        if meta["joint_names"] != JOINTS:
            raise ValueError("Joint order differs from this SO101 controller")
        if cfg["cond_steps"] != 1 or cfg["img_cond_steps"] != 1:
            raise ValueError("This controller supports one observation frame only")
        if cfg["action_dim"] != 6 or cfg["obs_dim"] != 6:
            raise ValueError("Expected six native SO101 joint values")
        self.device = torch.device(device)
        self.steps = steps
        self.fps = int(meta["fps"])
        self.shape = tuple(cfg["shape_meta"]["obs"]["rgb"]["shape"])
        if self.shape != tuple(meta["image_shape"]) or self.shape[0] != 3:
            raise ValueError("Config and manifest image shapes disagree")
        ncfg = cfg["model"]["network"]
        bcfg = ncfg["backbone"]
        backbone = VitEncoder(list(self.shape), SimpleNamespace(**bcfg["cfg"]),
                              num_channel=3, img_h=self.shape[1], img_w=self.shape[2])
        network_options = {k: v for k, v in ncfg.items() if k not in (
            "_target_", "backbone", "img_cond_steps", "cond_dim", "horizon_steps", "action_dim", "augment")}
        network = VisionFlowMLP(backbone=backbone, action_dim=6,
            horizon_steps=cfg["horizon_steps"], cond_dim=6, img_cond_steps=1,
            augment=False, **network_options)
        self.model = ReFlow(network, str(self.device), cfg["horizon_steps"], 6,
            cfg["model"]["act_min"], cfg["model"]["act_max"], 6, steps,
            cfg.get("seed", 42), sample_t_type=cfg["model"].get("sample_t_type", "uniform"))
        checkpoint = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint[weights], strict=True)
        self.model.eval()
        self.horizon = cfg["horizon_steps"]
        with np.load(stats, allow_pickle=False) as z:
            self.stats = {k: z[k].astype(np.float32) for k in (
                "obs_min", "obs_max", "action_min", "action_max")}
        for v in self.stats.values():
            if v.shape != (6,) or not np.isfinite(v).all():
                raise ValueError("Invalid normalization statistics")
        for prefix in ("obs", "action"):
            if np.any(self.stats[prefix+"_max"] <= self.stats[prefix+"_min"]):
                raise ValueError(f"Degenerate {prefix} normalization range")
        log.info("Loaded epoch %s, weights=%s, device=%s", checkpoint.get("epoch"), weights, device)

    def normalize_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError("Invalid joint observation")
        return 2*(state-self.stats["obs_min"])/(self.stats["obs_max"]-self.stats["obs_min"]+1e-6)-1

    @torch.inference_mode()
    def predict(self, rgb, state):
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Camera must return RGB uint8 HWC")
        # Exactly match prepare.py: RGB PIL bilinear resize, uint8 0..255.
        image = np.asarray(Image.fromarray(rgb).resize(
            (self.shape[2], self.shape[1]), Image.Resampling.BILINEAR))
        image = torch.from_numpy(image.transpose(2,0,1).copy())[None,None].to(self.device)
        state = torch.from_numpy(self.normalize_state(state))[None,None].to(self.device)
        result = self.model.sample(dict(rgb=image,state=state), inference_steps=self.steps,
                                   clip_intermediate_actions=True).trajectories[0].cpu().numpy()
        result = (result+1)/2*(self.stats["action_max"]-self.stats["action_min"]+1e-6)+self.stats["action_min"]
        if result.shape != (self.horizon,6) or not np.isfinite(result).all():
            raise ValueError("Model returned invalid actions")
        return result


def limited_target(target, current, previous, dt, fps, stats, arm_speed, gripper_speed):
    """Bound targets in training coordinates and slew commands in their native units."""
    target, current, previous = [np.asarray(v,dtype=np.float32) for v in (target,current,previous)]
    if any(v.shape != (6,) or not np.isfinite(v).all() for v in (target,current,previous)):
        raise ValueError("Non-finite or malformed target/state")
    dt = min(max(dt,0),2/fps)  # a delayed loop must not permit a large jump
    allowed = np.array([arm_speed]*5+[gripper_speed],dtype=np.float32)*dt
    # Training bounds constrain the policy's desired target, not the transition
    # from an already-outside starting pose. Hardware bounds are enforced in raw_action().
    target = np.clip(target,stats["action_min"],stats["action_max"])
    lower = np.maximum(previous-allowed,current-5)
    upper = np.minimum(previous+allowed,current+5)
    if np.any(lower > upper):
        details = "; ".join(
            f"{JOINTS[i]} measured={current[i]:.3f}, previous={previous[i]:.3f}, "
            f"allowed_step={allowed[i]:.3f}, tracking_limit=5"
            for i in np.flatnonzero(lower > upper))
        raise RuntimeError("Joint tracking and slew limits cannot be met: "+details)
    return np.clip(target,lower,upper)


class Preview:
    """Show frames via Tk/Pillow; never opens a second camera."""
    def __init__(self):
        import tkinter as tk
        from PIL import ImageTk
        self.ImageTk = ImageTk
        self.root = tk.Tk()
        self.root.title("SO101 policy — close window to stop")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _: self.close())

    def update(self, rgb):
        if self.closed:
            return False
        self.photo = self.ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.configure(image=self.photo)
        self.root.update()
        return not self.closed

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()


def check_servos(bus):
    """Individual acknowledged reads expose status errors that group reads can miss."""
    errors = []
    for name, motor in bus.motors.items():
        try:
            torque = bus.read("Torque_Enable", name)
            log.info("Servo %s (%s): status OK, torque=%s",motor.id,name,torque)
        except Exception as exc:
            errors.append(f"id={motor.id} {name}: {exc}")
    if errors:
        raise RuntimeError("Servo preflight failed; no goal positions written:\n"+"\n".join(errors))


def check_execution_health(bus):
    """Group position reads can hide status bits; acknowledged reads must check them."""
    for name,motor in bus.motors.items():
        try:
            torque = bus.read("Torque_Enable",name)
            if torque != 1:
                raise RuntimeError(f"Torque_Enable={torque}; torque was disabled unexpectedly")
        except Exception as exc:
            raise RuntimeError(f"Stopping execution: servo id={motor.id} ({name}) fault: {exc}") from exc


def close_bus(bus, disable_torque):
    """A fault on one motor must not prevent shutdown attempts on the others."""
    if not bus.is_connected:
        return
    if disable_torque:
        for name, motor in bus.motors.items():
            try:
                bus.disable_torque(name,num_retry=1)
            except Exception:
                log.exception("Could not confirm torque disabled for id=%s (%s)",motor.id,name)
    try:
        bus.disconnect(disable_torque=False)
    except Exception:
        log.exception("Error closing serial port")
        bus.port_handler.closePort()


def position_limits(bus):
    """Intersect hardware and calibration bounds; keep one encoder tick inside endpoints."""
    limits = {}
    for name in bus.motors:
        lo = int(bus.read("Min_Position_Limit",name,normalize=False))
        hi = int(bus.read("Max_Position_Limit",name,normalize=False))
        cal = bus.calibration[name]
        lo,hi = max(lo,cal.range_min)+1,min(hi,cal.range_max)-1
        if hi <= lo:
            raise RuntimeError(f"Invalid position limits for {name}: {lo}..{hi}")
        limits[name] = (lo,hi)
    return limits


def bounded_raw(raw, limits):
    result = {}
    for name,value in raw.items():
        lo,hi = limits[name]
        bounded = int(np.clip(int(value),lo,hi))
        if abs(bounded-int(value)) > 16:
            raise RuntimeError(f"{name}: raw target {value} is too far outside {lo}..{hi}")
        result[name] = bounded
    return result


def raw_action(bus, native, limits):
    # Final protection applies AFTER native-unit conversion and integer quantization.
    ids = {bus.motors[k.removesuffix(".pos")].id:float(v) for k,v in zip(JOINTS,native)}
    raw = bus._unnormalize(ids)
    raw = {name:raw[motor.id] for name,motor in bus.motors.items()}
    return bounded_raw(raw,limits)


def native_action(bus, raw):
    values = bus._normalize({bus.motors[n].id:v for n,v in raw.items()})
    return np.array([values[bus.motors[k.removesuffix(".pos")].id] for k in JOINTS],dtype=np.float64)


def diagnose(args):
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    robot = SOFollower(SOFollowerRobotConfig(port=args.port,id=args.robot_id,
        use_degrees=True,motor_model=getattr(args,"motor_model","hx30hm"),cameras={}))
    try:
        # A faulted motor may fail ping; still inspect all remaining motors individually.
        robot.bus.connect(handshake=False)
        check_servos(robot.bus)
    finally:
        close_bus(robot.bus,disable_torque=False)


def release(args):
    """Explicit recovery command: disable each motor without requiring a healthy handshake."""
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    robot = SOFollower(SOFollowerRobotConfig(port=args.port,id=args.robot_id,
        use_degrees=True,motor_model=getattr(args,"motor_model","hx30hm"),cameras={}))
    errors = []
    try:
        robot.bus.connect(handshake=False)
        for name,motor in robot.bus.motors.items():
            try:
                robot.bus.disable_torque(name,num_retry=1)
                enabled = robot.bus.read("Torque_Enable",name)
                if enabled != 0:
                    raise RuntimeError(f"Torque_Enable is still {enabled}")
                log.info("Servo %s (%s): torque OFF verified",motor.id,name)
            except Exception as exc:
                errors.append(f"id={motor.id} {name}: {exc}")
                log.error("Could not verify release of id=%s (%s): %s",motor.id,name,exc)
    finally:
        close_bus(robot.bus,disable_torque=False)
    if errors:
        raise RuntimeError("Some servos could not be verified released:\n"+"\n".join(errors))


def offline_check(policy, archive):
    """One saved observation, same preprocessing as live mode; no hardware imports."""
    with np.load(archive,allow_pickle=False) as z:
        rgb = z["images"][0].transpose(1,2,0)
        state = (z["states"][0]+1)/2*(policy.stats["obs_max"]-policy.stats["obs_min"]+1e-6)+policy.stats["obs_min"]
    result = policy.predict(rgb,state)
    print(json.dumps(dict(mode="offline",shape=list(result.shape),joint_names=JOINTS,
        units=["deg"]*5+["gripper_0_100"],first_action=result[0].tolist()),indent=2))


def live(args, policy):
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.configs import Cv2Backends
    robot = SOFollower(SOFollowerRobotConfig(port=args.port,id=args.robot_id,
                                            use_degrees=True,motor_model=getattr(args,"motor_model","hx30hm"),cameras={}))
    if not robot.calibration:
        raise RuntimeError(f"Calibration missing: {robot.calibration_fpath}; calibrate before inference")
    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=args.camera_index,width=640,height=480,
        fps=policy.fps,warmup_s=2,backend=Cv2Backends(getattr(args,"camera_backend",1200 if sys.platform=="darwin" else 0))))
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    old_signal = signal.signal(signal.SIGINT, request_stop)
    old_term = signal.signal(signal.SIGTERM, request_stop)
    preview = None
    motion_enabled = False
    limits = None
    started = time.perf_counter()
    previous = None
    previous_send = None
    next_health_check = 0
    last_print = 0
    log.info("Mode=%s; arm units=degrees, gripper=0..100", "EXECUTE" if args.execute else "READ-ONLY")
    try:
        # Bus.connect() only opens serial and pings; avoid robot.connect(), which configures torque.
        cam.connect()
        robot.bus.connect()
        if args.execute:
            check_servos(robot.bus)
            limits = position_limits(robot.bus)
        if not robot.is_calibrated:
            raise RuntimeError("Saved calibration differs from motors; run the calibration workflow first")
        if args.preview:
            preview = Preview()
        started = time.perf_counter()
        with args.log.open("w",buffering=1) as output:
            while not stop and time.perf_counter()-started < args.duration:
                tick = time.perf_counter()
                obs = robot.bus.sync_read("Present_Position")
                current = np.array([obs[k.removesuffix(".pos")] for k in JOINTS],dtype=np.float32)
                cam.read_latest(max_age_ms=500)
                with cam.frame_lock:
                    rgb = cam.latest_frame.copy()
                    captured_at = cam.latest_timestamp
                normalized = policy.normalize_state(current)
                if args.execute and np.abs(normalized).max() > args.max_normalized_state:
                    raise RuntimeError("Current pose lies outside the training range; inspect in read-only mode")
                if preview and not preview.update(rgb):
                    break
                if stop:
                    break
                predict_start = time.perf_counter()
                actions = policy.predict(rgb,current)
                inference_ms = (time.perf_counter()-predict_start)*1000
                if stop:
                    break
                # Check freshness again after inference. Do not act on an expired observation.
                if captured_at is None or (time.perf_counter()-captured_at) > .5:
                    raise TimeoutError("Observation expired during inference (>500ms); no action sent")
                cam.read_latest(max_age_ms=500)
                target = actions[0]
                applied = None
                if args.execute:
                    if not motion_enabled:
                        # Hold the measured pose before official configuration re-enables torque.
                        # Read raw ticks directly: float32 degrees can round endpoint 905 to 904.
                        hold = {name:int(robot.bus.read("Present_Position",name,normalize=False))
                                for name in robot.bus.motors}
                        hold = bounded_raw(hold,limits)
                        motion_enabled = True
                        for name,value in hold.items():
                            robot.bus.write("Goal_Position",name,value,normalize=False)
                        robot.configure()
                        previous = native_action(robot.bus,hold)
                        previous_send = time.perf_counter()-1/policy.fps
                    now = time.perf_counter()
                    if now >= next_health_check:
                        check_execution_health(robot.bus)
                        next_health_check = time.perf_counter()+0.25
                    # Re-read the joints rather than applying a target relative to a stale pose.
                    measured = robot.bus.sync_read("Present_Position")
                    current = np.array([measured[k.removesuffix(".pos")] for k in JOINTS],dtype=np.float32)
                    if stop:
                        break
                    if time.perf_counter()-captured_at > .5:
                        raise TimeoutError("Observation expired before sending action; stopping")
                    applied = limited_target(target,current,previous,now-previous_send,policy.fps,
                        policy.stats,args.arm_speed,args.gripper_speed)
                    raw = raw_action(robot.bus,applied,limits)
                    robot.bus.sync_write("Goal_Position",raw,normalize=False)
                    previous = native_action(robot.bus,raw)
                    applied = previous.copy()
                    previous_send = time.perf_counter()
                row = dict(elapsed_s=round(time.perf_counter()-started,3),execute=args.execute,
                    inference_ms=round(inference_ms,2),state=current.tolist(),
                    target=target.tolist(),applied=None if applied is None else applied.tolist())
                output.write(json.dumps(row)+"\n")
                if tick-last_print >= 1:
                    log.info("infer %.1fms | state %s | predicted %s",inference_ms,
                        np.round(current,1),np.round(target,1))
                    last_print = tick
                time.sleep(max(0,1/policy.fps-(time.perf_counter()-tick)))
    finally:
        # Release camera even if a servo reports an error while disabling torque.
        if robot.bus.is_connected:
            close_bus(robot.bus,disable_torque=motion_enabled)
        if cam.is_connected or cam.thread is not None:
            cam.disconnect()
        if preview:
            preview.close()
        signal.signal(signal.SIGINT,old_signal)
        signal.signal(signal.SIGTERM,old_term)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,default=ROOT/"checkpoints/best_val.pt")
    p.add_argument("--config",type=Path,default=DATA/"config.yaml")
    p.add_argument("--stats",type=Path,default=DATA/"normalization.npz")
    p.add_argument("--manifest",type=Path,default=DATA/"manifest.json")
    p.add_argument("--device",choices=["cpu","mps"],default="cpu")
    p.add_argument("--weights",choices=["model","ema"],default="model")
    p.add_argument("--steps",type=int,default=20)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute",dest="execute",action="store_true",help="Enable torque and send model actions")
    mode.add_argument("--read-only",dest="execute",action="store_false",help="Read observations and predict without motor writes")
    p.set_defaults(execute=None)
    p.add_argument("--diagnose",action="store_true",help="Read individual servo status only; no camera or motion")
    p.add_argument("--release",action="store_true",help="Disable torque on each servo and verify; support the arm first")
    preview_mode = p.add_mutually_exclusive_group()
    preview_mode.add_argument("--preview",dest="preview",action="store_true",help="Display the same captured frames (default)")
    preview_mode.add_argument("--no-preview",dest="preview",action="store_false",help="Disable the camera preview window")
    p.set_defaults(preview=True)
    p.add_argument("--offline-data",type=Path,help="Check a converted NPZ without connecting hardware")
    p.add_argument("--port",default=None)
    p.add_argument("--robot-id",default="follower")
    p.add_argument("--motor-model",choices=["hx30hm","sts3215"],default="hx30hm")
    p.add_argument("--camera-backend",type=int,default=1200 if sys.platform=="darwin" else 0)
    p.add_argument("--run",type=Path,help="Training result directory containing checkpoint/ and metadata")
    p.add_argument("--camera-index",type=int,default=0)
    p.add_argument("--duration",type=float,default=30)
    p.add_argument("--arm-speed",type=float,default=30,help="Max joint target slew, degrees/s")
    p.add_argument("--gripper-speed",type=float,default=30,help="Max gripper target slew, percentage points/s")
    p.add_argument("--max-normalized-state",type=float,default=1.5)
    p.add_argument("--log",type=Path,default=PROJECT/"logs"/f"policy_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    args = p.parse_args()
    if args.run:
        args.checkpoint = args.run/"checkpoint/best_val.pt"
        args.config = args.run/"config.yaml"
        args.stats = args.run/"normalization.npz"
        args.manifest = args.run/"manifest.json"
    if any(not np.isfinite(v) or v <= 0 for v in (
        args.steps,args.duration,args.arm_speed,args.gripper_speed,args.max_normalized_state)):
        p.error("Steps, duration, speed limits and state threshold must be positive and finite")
    if args.offline_data and args.execute:
        p.error("--offline-data cannot be combined with --execute")
    if args.diagnose and (args.execute or args.offline_data):
        p.error("--diagnose cannot be combined with execution or offline inference")
    if args.release and (args.execute or args.offline_data or args.diagnose):
        p.error("--release cannot be combined with another mode")
    if args.execute is None:
        args.execute = False
    if not args.offline_data and not args.port:
        p.error("--port is required for live observations, diagnosis, release or execution")
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    if args.release:
        release(args)
        return
    if args.diagnose:
        diagnose(args)
        return
    policy = Policy(args.checkpoint,args.config,args.stats,args.manifest,args.device,args.weights,args.steps)
    if args.offline_data:
        offline_check(policy,args.offline_data)
    else:
        args.log.parent.mkdir(parents=True,exist_ok=True)
        live(args,policy)


if __name__ == "__main__":
    main()
