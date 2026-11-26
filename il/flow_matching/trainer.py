"""Offline SO101 trainer using the official ReinFlow model and FM objective.

No simulator, online RL or WandB dependency. See LICENSE-ReinFlow for upstream
model/loader licensing; this runner adapts the official pretraining routine.
"""
import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import StitchedSequenceDataset
from .model import build_model
from .scheduler import CosineAnnealingWarmupRestarts

log = logging.getLogger(__name__)


class FlowMatchingTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(4)
        self.model = build_model(cfg)
        self.ema_model = copy.deepcopy(self.model)
        datasets = []
        for data in (cfg.train_dataset, cfg.val_dataset):
            datasets.append(StitchedSequenceDataset(
                dataset_path=data.dataset_path, horizon_steps=data.horizon_steps,
                cond_steps=data.cond_steps, img_cond_steps=data.img_cond_steps,
                max_n_episodes=data.max_n_episodes, use_img=data.use_img,
                device=data.device,
            ))
        self.dataset_train, self.dataset_val = datasets
        if any(dataset.states.shape[1] != cfg.model.obs_dim for dataset in datasets):
            raise ValueError('Dataset/model observation dimension mismatch')
        # Conversion puts disjoint episodes in distinct files before normalization.
        manifest = json.loads((Path(cfg.train_dataset_path).parent/"manifest.json").read_text())
        assert not set(manifest["train_episodes"]) & set(manifest["val_episodes"])
        self.dataloader_train = DataLoader(self.dataset_train,batch_size=cfg.train.batch_size,
                                           shuffle=True,num_workers=0,drop_last=True)
        self.dataloader_val = DataLoader(self.dataset_val,batch_size=cfg.train.batch_size,
                                         shuffle=False,num_workers=0,drop_last=False)
        if not len(self.dataloader_train) or not len(self.dataloader_val):
            raise ValueError("No training/validation batches; check episode lengths and reduce batch size")
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=cfg.train.learning_rate,
                                           weight_decay=cfg.train.weight_decay)
        schedule = cfg.train.lr_scheduler
        self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
            first_cycle_steps=schedule.first_cycle_steps,cycle_mult=1.0,
            max_lr=cfg.train.learning_rate,min_lr=schedule.min_lr,
            warmup_steps=schedule.warmup_steps,gamma=1.0)
        self.output = Path(cfg.logdir)
        self.checkpoints = self.output/"checkpoint"
        self.checkpoints.mkdir(parents=True,exist_ok=True)
        self.global_step = 0
        self.start_epoch = 1
        self.best_val = float("inf")
        if cfg.get("base_policy_path"):
            c = torch.load(cfg.base_policy_path,map_location=self.device,weights_only=True)
            self.model.load_state_dict(c["model"])
            self.ema_model.load_state_dict(c["ema"])
            self.optimizer.load_state_dict(c["optimizer"])
            self.scheduler.load_state_dict(c["lr_scheduler"])
            self.start_epoch = int(c["epoch"])+1
            self.global_step = int(c.get("global_step",0))
            self.best_val = float(c.get("best_val",float("inf")))
        (self.output/"architecture.log").write_text(str(self.model))
        log.info("Train windows=%s, validation windows=%s, updates/epoch=%s",
                 len(self.dataset_train),len(self.dataset_val),len(self.dataloader_train))

    def objective(self, batch):
        actions = batch.actions.to(self.device)
        cond = {k:v.to(self.device) for k,v in batch.conditions.items()}
        (xt,t),velocity = self.model.generate_target(actions)
        return self.model.loss(xt,t,cond,velocity)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        augment = self.model.network.augment
        self.model.network.augment = False
        total = count = 0
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type=="cuda" else []
        # Fixed evaluation noise/time; restore RNG so validation does not alter training.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(12345)
            for batch in self.dataloader_val:
                value = self.objective(batch)
                if not torch.isfinite(value):
                    raise FloatingPointError("Non-finite validation loss")
                n = len(batch.actions)
                total += value.item()*n
                count += n
        self.model.network.augment = augment
        return total/count

    def save(self, name, epoch):
        state = dict(epoch=epoch,global_step=self.global_step,best_val=self.best_val,
                     model=self.model.state_dict(),ema=self.ema_model.state_dict(),
                     optimizer=self.optimizer.state_dict(),lr_scheduler=self.scheduler.state_dict())
        tmp = self.checkpoints/(name+".tmp")
        torch.save(state,tmp)
        tmp.replace(self.checkpoints/name)

    def run(self):
        for epoch in range(self.start_epoch,self.start_epoch+self.cfg.train.n_epochs):
            self.model.train()
            total = count = 0
            for batch in self.dataloader_train:
                self.optimizer.zero_grad(set_to_none=True)
                value = self.objective(batch)
                if not torch.isfinite(value):
                    raise FloatingPointError("Non-finite training loss")
                value.backward()
                self.optimizer.step()
                self.global_step += 1
                if self.global_step % self.cfg.train.update_ema_freq == 0:
                    with torch.no_grad():
                        if epoch < self.cfg.train.epoch_start_ema:
                            self.ema_model.load_state_dict(self.model.state_dict())
                        else:
                            decay = self.cfg.ema.decay
                            for a,b in zip(self.ema_model.parameters(),self.model.parameters()):
                                a.lerp_(b,1-decay)
                n = len(batch.actions)
                total += value.item()*n
                count += n
            self.scheduler.step()
            val = self.validate() if epoch % self.cfg.train.val_freq==0 else None
            improved = val is not None and val < self.best_val
            if improved:
                self.best_val = val
            self.save("last.pt",epoch)
            if improved:
                self.save("best_val.pt",epoch)
            if epoch % self.cfg.train.save_model_freq==0 or epoch==self.start_epoch+self.cfg.train.n_epochs-1:
                self.save(f"state_{epoch}.pt",epoch)
            row = dict(epoch=epoch,global_step=self.global_step,train_loss=total/count,
                       val_loss=val,best_val=self.best_val)
            with (self.output/"metrics.jsonl").open("a") as f:
                f.write(json.dumps(row)+"\n")
            log.info("%s",json.dumps(row))
