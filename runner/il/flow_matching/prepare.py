"""LeRobot v3 -> ReinFlow NPZ; split episodes before fitting normalization."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-episodes", type=int, default=3)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    root = args.dataset.resolve()
    info = json.loads((root / "meta/info.json").read_text())
    data = pd.concat([pd.read_parquet(f) for f in sorted((root / "data").rglob("*.parquet"))])
    data = data.sort_values("index").reset_index(drop=True)
    meta = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta/episodes").rglob("*.parquet"))])
    meta = meta.sort_values("episode_index")
    assert len(data) == info["total_frames"] == int(meta.length.sum())
    assert np.array_equal(data["index"], np.arange(len(data)))
    states = np.stack(data["observation.state"]).astype(np.float32)
    actions = np.stack(data.action).astype(np.float32)
    assert states.shape == actions.shape == (len(data), 6)
    assert np.isfinite(states).all() and np.isfinite(actions).all()
    episode = data.episode_index.to_numpy()
    ids = np.random.default_rng(args.seed).permutation(meta.episode_index.to_numpy())
    assert 0 < args.val_episodes < len(ids)
    val_ids = sorted(ids[:args.val_episodes].tolist())
    train_ids = sorted(ids[args.val_episodes:].tolist())
    train_mask = np.isin(episode, train_ids)
    stats = dict(obs_min=states[train_mask].min(0), obs_max=states[train_mask].max(0),
                 action_min=actions[train_mask].min(0), action_max=actions[train_mask].max(0))
    normalized_states = 2 * (states-stats["obs_min"]) / (stats["obs_max"]-stats["obs_min"]+1e-6)-1
    normalized_actions = 2 * (actions-stats["action_min"]) / (stats["action_max"]-stats["action_min"]+1e-6)-1
    # Keep held-out values outside the training range: do not conceal extrapolation.
    images = np.empty((len(data), 3, 96, 96), dtype=np.uint8)
    assigned = np.zeros(len(data), dtype=bool)
    sources = {}
    key = "observation.images.wrist"
    for _, e in meta.iterrows():
        ix = np.flatnonzero(episode == e.episode_index)
        assert len(ix) == int(e.length) and np.all(np.diff(ix) == 1)
        assert ix[0] == int(e.dataset_from_index) and ix[-1]+1 == int(e.dataset_to_index)
        assert np.allclose(data.loc[ix, "timestamp"], np.arange(len(ix))/info["fps"], atol=1e-5)
        prefix = f"videos/{key}"
        src = root / info["video_path"].format(video_key=key,
            chunk_index=int(e[prefix+"/chunk_index"]), file_index=int(e[prefix+"/file_index"]))
        start = round(e[prefix+"/from_timestamp"]*info["fps"])
        mapping = sources.setdefault(src, {})
        for j, dest in enumerate(ix):
            assert start+j not in mapping
            mapping[start+j] = dest
    for src, mapping in sources.items():
        with av.open(str(src)) as c:
            for frame_idx, frame in enumerate(c.decode(video=0)):
                if frame_idx in mapping:
                    dest = mapping[frame_idx]
                    image = frame.to_image().resize((96,96), Image.Resampling.BILINEAR)
                    images[dest] = np.asarray(image).transpose(2,0,1)
                    assigned[dest] = True
    assert assigned.all(), "Missing video frames"
    args.out.mkdir(parents=True)
    np.savez(args.out/"normalization.npz", **stats)
    manifest = dict(source=str(root), seed=args.seed, fps=info["fps"],
        train_episodes=train_ids, val_episodes=val_ids, image_shape=[3,96,96],
        image_transform="RGB bilinear resize to 96x96, uint8 0..255",
        joint_names=info["features"]["action"]["names"],
        action_representation="absolute_joint_positions_in_lerobot_native_units",
        joint_units=["degrees"]*5+["calibrated_0_100"], splits={})
    for split, selected in (("train",train_ids),("val",val_ids)):
        mask = np.isin(episode,selected)
        lengths = np.array([np.sum(episode==i) for i in selected],dtype=np.int64)
        assert (lengths >= 16).all()
        np.savez_compressed(args.out/f"{split}.npz", states=normalized_states[mask],
            actions=normalized_actions[mask], images=images[mask], traj_lengths=lengths)
        manifest["splits"][split] = dict(frames=int(mask.sum()), trajectories=len(selected),
            windows_horizon16=int((lengths-15).sum()))
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
