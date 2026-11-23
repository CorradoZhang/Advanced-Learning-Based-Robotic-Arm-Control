# Data, training and evaluation

## Data contract

The converter reads a LeRobot v3 directory containing `meta/info.json`, episode
metadata Parquet, data Parquet and wrist-camera MP4 files. It checks frame counts,
episode indexes, video alignment and finite six-dimensional states/actions.
Keep only the demonstrations you intend to train on in the selected dataset.

Prepared archives contain:

| Key | Shape | Meaning |
|---|---|---|
| `states` | N×6 | normalized observed joint positions |
| `actions` | N×6 | normalized demonstrated joint targets |
| `images` | N×3×96×96 | RGB uint8, 0–255 |
| `traj_lengths` | number of episodes | contiguous episode boundaries |

The split uses whole episodes. Normalization statistics are fitted only on
training episodes. Held-out values outside that range are not clipped during
conversion. The loader never builds an action chunk across an episode boundary.
For horizon 16, a trajectory of L frames produces L−15 windows; final incomplete
chunks are omitted. Short trajectories produce no complete windows.

## Flow Matching objective

For demonstrated action chunk A and Gaussian noise ε, sample t uniformly in
[0,1] and construct x(t)=(1−t)ε+tA. The official-style velocity network receives
the image/state condition and regresses A−ε with mean squared error.
Inference integrates the learned velocity from noise using Euler steps.

`horizon_steps=16` is the action chunk length. `denoising_steps=20` is the number
of integration steps for generation; it is not the number of optimizer updates.
With batch size B and W training windows, each epoch has floor(W/B) updates
because the training loader drops its final incomplete batch. Validation keeps
its final partial batch and weights loss by sample count.

## Run and resume

```bash
python -m runner.flow_matching.train --data data/pick_place \
  --epochs 200 --batch-size 64 --device cuda:0

# Resume the same dataset for 20 additional epochs, into a new output directory:
python -m runner.flow_matching.train --data data/pick_place \
  --resume result/flow_matching/YOUR_RUN/checkpoint/last.pt --epochs 20 --device cuda:0
```

The run directory includes the actual model settings, normalization and split
manifest. Resume checks the statistics/split instead of silently mixing a
changed dataset with an old model's coordinate system. A changed dataset should
start a new experiment. Output directories are not overwritten.

Set `--data-device cpu` (default) to avoid storing all images in GPU memory.
Use `--data-device cuda:0` only if sufficient memory is available. `--device cpu`
can run the smoke test and small experiments. Select a smaller batch if needed.

## Checkpoint use

`best_val.pt` is selected using raw-model FM validation loss; `model` is the
corresponding state-dict key. `ema` is also saved and can be selected explicitly
for separate evaluation. A lower loss does not guarantee a higher task success
rate. Keep task-level evaluation separate from random-window action error.

Deploy with the checkpoint's own `config.yaml`, `normalization.npz` and
`manifest.json`, and matching robot calibration. Do not combine a new checkpoint
with a previous dataset's statistics, camera mounting or action units.

## Reproducibility and current scope

The tests create a synthetic three-episode video/Parquet fixture and exercise
conversion, a small CPU training run, checkpoint reload and offline prediction.
This verifies software/data plumbing, not robot learning performance. The project
does not ship demonstration data or claim a benchmark success rate. VLA,
language-conditioned tasks, online RL and sim-to-real training are not implemented
by these entry points.
