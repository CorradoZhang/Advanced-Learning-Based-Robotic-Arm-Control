"""Portable offline Flow Matching training entry point."""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True,help="Prepared dataset directory")
    p.add_argument("--config",type=Path,default=ROOT/"configs/flow_matching.yaml")
    p.add_argument("--output",type=Path,help="New output directory; existing runs are never overwritten")
    p.add_argument("--epochs",type=int,default=200)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--data-device",default="cpu")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--resume",type=Path,help="Checkpoint from the same prepared dataset")
    args = p.parse_args()
    if args.epochs<1 or args.batch_size<1:
        p.error("epochs and batch size must be positive")
    from omegaconf import OmegaConf
    import numpy as np
    from algorithm.flow_matching.trainer import FlowMatchingTrainer
    data = args.data.resolve()
    for name in ("train.npz","val.npz","normalization.npz","manifest.json"):
        if not (data/name).is_file():
            p.error(f"Missing {data/name}; run the data converter first")
    if args.resume:
        old = args.resume.resolve().parent.parent
        if not (old/"normalization.npz").is_file():
            p.error("Resume requires the checkpoint's original run metadata")
        with np.load(old/"normalization.npz") as a,np.load(data/"normalization.npz") as b:
            if any(not np.array_equal(a[k],b[k]) for k in b.files):
                p.error("Resume normalization differs; start a new run for a changed dataset")
        previous=json.loads((old/"manifest.json").read_text())
        current=json.loads((data/"manifest.json").read_text())
        for key in ("train_episodes","val_episodes","splits","joint_names"):
            if previous[key] != current[key]:
                p.error(f"Resume dataset differs: {key}")
    output=args.output or ROOT/"result/flow_matching"/f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.seed}"
    if output.exists():
        p.error(f"Output already exists: {output}")
    cfg=OmegaConf.load(args.config)
    cfg.device=cfg.model.device=args.device
    cfg.seed=cfg.model.seed=args.seed
    cfg.logdir=str(output.resolve())
    cfg.train_dataset_path=str(data/"train.npz")
    cfg.base_policy_path=str(args.resume.resolve()) if args.resume else None
    cfg.train.n_epochs=args.epochs
    cfg.train.batch_size=args.batch_size
    cfg.train.lr_scheduler.first_cycle_steps=max(2,args.epochs)
    cfg.train.lr_scheduler.warmup_steps=min(5,max(2,args.epochs)-1)
    cfg.train_dataset.dataset_path=str(data/"train.npz")
    cfg.val_dataset.dataset_path=str(data/"val.npz")
    cfg.train_dataset.device=cfg.val_dataset.device=args.data_device
    output.mkdir(parents=True)
    OmegaConf.save(cfg,output/"config.yaml")
    for name in ("normalization.npz","manifest.json"):
        shutil.copy2(data/name,output/name)
    logging.basicConfig(level=logging.INFO,handlers=[logging.StreamHandler(),logging.FileHandler(output/"train.log")])
    FlowMatchingTrainer(cfg).run()
    print(f"Outputs saved to {output.resolve()}")


if __name__=="__main__":
    main()
