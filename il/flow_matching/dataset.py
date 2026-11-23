"""Episode-bounded action windows from normalized SO101 NPZ data.

Adapted from ReinFlow e722e151bed767f3ffef47527cf697f2358af55d.
See LICENSE-ReinFlow for original licensing.
"""
from collections import namedtuple

import numpy as np
import torch

Batch = namedtuple("Batch", "actions conditions")


class StitchedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, horizon_steps=16, cond_steps=1,
                 img_cond_steps=1, max_n_episodes=-1, use_img=True, device="cuda:0"):
        if not 1 <= img_cond_steps <= cond_steps or horizon_steps < 1:
            raise ValueError("Require positive horizon and 1 <= img_cond_steps <= cond_steps")
        if max_n_episodes != -1 and max_n_episodes < 1:
            raise ValueError("max_n_episodes must be -1 or positive")
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.use_img = use_img
        with np.load(dataset_path, allow_pickle=False) as data:
            lengths = data["traj_lengths"]
            if lengths.ndim!=1 or not np.issubdtype(lengths.dtype,np.integer) or np.any(lengths<=0):
                raise ValueError("traj_lengths must be positive integer episode lengths")
            n=int(lengths.sum())
            if data["states"].shape!=(n,6) or data["actions"].shape!=(n,6):
                raise ValueError("Expected states/actions [sum(traj_lengths), 6]")
            if not np.isfinite(data["states"]).all() or not np.isfinite(data["actions"]).all():
                raise ValueError("Non-finite states/actions")
            if use_img and (data["images"].shape!=(n,3,96,96) or data["images"].dtype!=np.uint8):
                raise ValueError("Expected RGB images [N,3,96,96], uint8 0..255")
            if max_n_episodes != -1:
                lengths = lengths[:max_n_episodes]
            steps = int(lengths.sum())
            self.states = torch.from_numpy(data["states"][:steps]).float().to(device)
            self.actions = torch.from_numpy(data["actions"][:steps]).float().to(device)
            if use_img:
                self.images = torch.from_numpy(data["images"][:steps]).to(device)
        self.indices = []
        offset = 0
        for length in lengths:
            self.indices.extend((start, start - offset)
                                for start in range(offset, offset + length - horizon_steps + 1))
            offset += length

    def __getitem__(self, index):
        start, history = self.indices[index]
        # Repeat the first observation when history reaches an episode boundary.
        state_indices = [start - min(t, history) for t in reversed(range(self.cond_steps))]
        conditions = {"state": torch.stack([self.states[i] for i in state_indices])}
        if self.use_img:
            image_indices = [start - min(t, history) for t in reversed(range(self.img_cond_steps))]
            conditions["rgb"] = torch.stack([self.images[i] for i in image_indices])
        return Batch(self.actions[start:start + self.horizon_steps], conditions)

    def __len__(self):
        return len(self.indices)
