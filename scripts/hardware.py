"""Calibration, teleoperation and LeRobot recording with explicit hardware settings."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]


def build_command(args, hardware):
    def setting(name,default=None):
        return getattr(args,name,None) if getattr(args,name,None) is not None else hardware.get(name,default)
    model=setting("motor_model","hx30hm")
    if model not in ("hx30hm","sts3215"):
        raise ValueError("Supported motor_model values: hx30hm, sts3215")
    def arm(prefix):
        role="leader" if prefix=="teleop" else "follower"
        port=setting(role+"_port")
        if not port:
            raise ValueError(f"Set {role}_port in your hardware config or --{role}-port")
        return [f"--{prefix}.type=so101_{role}",f"--{prefix}.port={port}",
                f"--{prefix}.id={setting(role+'_id',role)}",f"--{prefix}.motor_model={model}",
                f"--{prefix}.use_degrees=true"]
    if args.command.startswith("calibrate"):
        prefix="teleop" if args.command=="calibrate-leader" else "robot"
        return [sys.executable,"-m","lerobot.scripts.lerobot_calibrate",*arm(prefix)]
    if args.command=="teleoperate":
        return [sys.executable,"-m","lerobot.scripts.lerobot_teleoperate",*arm("robot"),*arm("teleop")]
    if args.episodes<1:
        raise ValueError("episodes must be positive; in resume mode this is the number to append")
    if args.resume:
        destination=args.resume.resolve()
        if not (destination/"meta/info.json").is_file():
            raise ValueError(f"Existing dataset metadata not found: {destination}")
        resume=True
    else:
        name=args.name or f"so101_pick_place_{datetime.now():%Y%m%d_%H%M%S_%f}"
        if not re.fullmatch(r"[A-Za-z0-9_-]+",name):
            raise ValueError("Dataset name must contain only letters, digits, underscores or hyphens")
        destination=args.dataset_root.resolve()/name
        if destination.exists():
            raise ValueError(f"Dataset exists; use --resume to append: {destination}")
        resume=False
    camera=dict(type="opencv",index_or_path=setting("camera_index",0),width=640,height=480,
                fps=30,backend=setting("camera_backend",1200 if sys.platform=="darwin" else 0),warmup_s=2)
    return [sys.executable,str(ROOT/"scripts/record.py"),f"--resume={str(resume).lower()}",
            *arm("robot"),*arm("teleop"),f"--robot.cameras={json.dumps({'wrist':camera})}",
            f"--dataset.repo_id=local/{destination.name}",f"--dataset.root={destination}",
            f"--dataset.single_task={args.task}",f"--dataset.num_episodes={args.episodes}",
            "--dataset.fps=30","--dataset.episode_time_s=30","--dataset.reset_time_s=15",
            "--dataset.vcodec=h264","--dataset.num_image_writer_threads_per_camera=1",
            "--dataset.push_to_hub=false",f"--display_data={str(not args.no_display).lower()}"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,help="Local hardware YAML; see configs/hardware.example.yaml")
    p.add_argument("--leader-port")
    p.add_argument("--follower-port")
    p.add_argument("--motor-model",choices=["hx30hm","sts3215"])
    p.add_argument("--print-command",action="store_true",help="Print only; do not access hardware")
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("calibrate-leader","calibrate-follower","teleoperate"):
        sub.add_parser(name)
    record=sub.add_parser("record")
    record.add_argument("--episodes",type=int,default=20)
    record.add_argument("--task",required=True)
    record.add_argument("--name")
    record.add_argument("--dataset-root",type=Path,default=ROOT/"datasets")
    record.add_argument("--resume",type=Path)
    record.add_argument("--camera-index",type=int)
    record.add_argument("--no-display",action="store_true")
    args=p.parse_args()
    try:
        hardware=yaml.safe_load(args.config.read_text()) if args.config else {}
        cmd=build_command(args,hardware or {})
    except (ValueError,OSError) as exc:
        p.error(str(exc))
    print(shlex.join(cmd),flush=True)
    if not args.print_command:
        result=subprocess.run(cmd,cwd=ROOT)
        raise SystemExit(result.returncode)


if __name__=="__main__":
    main()
