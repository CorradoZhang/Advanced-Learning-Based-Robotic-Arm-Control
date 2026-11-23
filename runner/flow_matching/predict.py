"""Offline validation prediction; never imports camera or motor drivers."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True,help="Prepared val.npz")
    p.add_argument("--index",type=int,default=0)
    p.add_argument("--device",default="cpu")
    p.add_argument("--weights",choices=["model","ema"],default="model")
    args=p.parse_args()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from algorithm.flow_matching.model import build_model
    cfg=OmegaConf.load(args.run/"config.yaml")
    cfg.device=args.device
    model=build_model(cfg)
    checkpoint=torch.load(args.run/"checkpoint/best_val.pt",map_location=args.device,weights_only=True)
    model.load_state_dict(checkpoint[args.weights],strict=True)
    model.eval()
    model.network.augment=False
    with np.load(args.data,allow_pickle=False) as data:
        if not 0<=args.index<len(data["states"]):
            p.error("Sample index is outside the dataset")
        if cfg.train_dataset.cond_steps!=1 or cfg.train_dataset.img_cond_steps!=1:
            p.error("This preview supports one observation frame")
        i=args.index
        cond={"state":torch.from_numpy(data["states"][i:i+1,None]).to(args.device),
              "rgb":torch.from_numpy(data["images"][i:i+1,None]).to(args.device)}
    with torch.inference_mode():
        prediction=model.sample(cond,inference_steps=cfg.denoising_steps).trajectories.cpu().numpy()[0]
    with np.load(args.run/"normalization.npz") as stats:
        actions=(prediction+1)/2*(stats["action_max"]-stats["action_min"]+1e-6)+stats["action_min"]
    if not np.isfinite(actions).all():
        raise ValueError("Non-finite prediction")
    meta=json.loads((args.run/"manifest.json").read_text())
    print(json.dumps(dict(shape=list(actions.shape),joint_names=meta["joint_names"],
        joint_units=meta["joint_units"],actions=actions.tolist()),indent=2))


if __name__=="__main__":
    main()
