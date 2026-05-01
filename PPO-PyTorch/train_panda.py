from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PPO import PPO
from panda_cube_grasp import PandaCubeGraspEnv, PandaCubeGraspIKEnv


ENV_IDS = {
    "joint": "PandaCubeGrasp-v0",
    "ik": "PandaCubeGraspIK-v0",
}


@dataclass
class TrainConfig:
    env: str = "ik"
    total_timesteps: int = 500_000
    max_ep_len: int = 200
    frame_skip: int = 5
    cube_xy_jitter: float = 0.01
    print_freq: int = 5_000
    log_freq: int = 1_000
    save_freq: int = 20_000
    eval_freq: int = 10_000
    eval_episodes: int = 10
    update_timestep: int = 1_600
    k_epochs: int = 40
    eps_clip: float = 0.2
    gamma: float = 0.99
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    action_std: float = 0.35
    action_std_decay_rate: float = 0.02
    min_action_std: float = 0.10
    action_std_decay_freq: int = 50_000
    seed: int = 0


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train PPO on PandaCubeGrasp environments.")
    parser.add_argument("--env", choices=tuple(ENV_IDS.keys()), default="ik")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--max-ep-len", type=int, default=200)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--cube-xy-jitter", type=float, default=0.01)
    parser.add_argument("--print-freq", type=int, default=5_000)
    parser.add_argument("--log-freq", type=int, default=1_000)
    parser.add_argument("--save-freq", type=int, default=20_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--update-timestep", type=int, default=1_600)
    parser.add_argument("--k-epochs", type=int, default=40)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--action-std", type=float, default=0.35)
    parser.add_argument("--action-std-decay-rate", type=float, default=0.02)
    parser.add_argument("--min-action-std", type=float, default=0.1)
    parser.add_argument("--action-std-decay-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    return TrainConfig(
        env=args.env,
        total_timesteps=args.total_timesteps,
        max_ep_len=args.max_ep_len,
        frame_skip=args.frame_skip,
        cube_xy_jitter=args.cube_xy_jitter,
        print_freq=args.print_freq,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        update_timestep=args.update_timestep,
        k_epochs=args.k_epochs,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        action_std=args.action_std,
        action_std_decay_rate=args.action_std_decay_rate,
        min_action_std=args.min_action_std,
        action_std_decay_freq=args.action_std_decay_freq,
        seed=args.seed,
    )


def get_env_id(config: TrainConfig) -> str:
    return ENV_IDS[config.env]


def build_env(config: TrainConfig) -> PandaCubeGraspEnv | PandaCubeGraspIKEnv:
    env_cls = PandaCubeGraspIKEnv if config.env == "ik" else PandaCubeGraspEnv
    return env_cls(
        render_mode=None,
        frame_skip=config.frame_skip,
        max_episode_steps=config.max_ep_len,
        cube_xy_jitter=config.cube_xy_jitter,
    )


def evaluate_policy(agent: PPO, config: TrainConfig) -> tuple[float, float]:
    eval_env = build_env(config)
    rewards = []
    successes = 0

    try:
        for episode in range(config.eval_episodes):
            state, _ = eval_env.reset(seed=config.seed + 10_000 + episode)
            episode_reward = 0.0

            for _ in range(config.max_ep_len):
                action = select_greedy_action(agent, state)
                state, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                if terminated:
                    successes += 1
                if terminated or truncated:
                    break

            rewards.append(episode_reward)
    finally:
        eval_env.close()

    return float(np.mean(rewards)), successes / float(config.eval_episodes)


def select_greedy_action(agent: PPO, state: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.policy_old.actor[0].weight.device)
        action_mean = agent.policy_old.actor(state_tensor)
    return action_mean.detach().cpu().numpy().flatten()


def next_run_paths(base_dir: Path, env_id: str) -> tuple[int, Path, Path]:
    log_dir = base_dir / "PPO_logs" / env_id
    ckpt_dir = base_dir / "PPO_preTrained" / env_id
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    current_num_files = [p for p in log_dir.glob("PPO_*_log_*.csv") if p.is_file()]
    run_num = len(current_num_files)

    log_path = log_dir / f"PPO_{env_id}_log_{run_num}.csv"
    metadata_path = ckpt_dir / f"PPO_{env_id}_run_{run_num}_metadata.json"
    return run_num, log_path, metadata_path


def train() -> None:
    config = parse_args()
    base_dir = Path(__file__).resolve().parent
    env_id = get_env_id(config)
    run_num, log_path, metadata_path = next_run_paths(base_dir, env_id)
    checkpoint_dir = base_dir / "PPO_preTrained" / env_id
    last_checkpoint_path = checkpoint_dir / f"PPO_{env_id}_{config.seed}_{run_num}.pth"
    best_checkpoint_path = checkpoint_dir / f"PPO_{env_id}_{config.seed}_{run_num}_best.pth"

    if config.seed:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    env = build_env(config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(
        state_dim,
        action_dim,
        config.lr_actor,
        config.lr_critic,
        config.gamma,
        config.k_epochs,
        config.eps_clip,
        has_continuous_action_space=True,
        action_std_init=config.action_std,
    )

    print("============================================================================================")
    print(f"Training environment : {env_id}")
    print(f"Control space        : {config.env}")
    print(f"Run number           : {run_num}")
    print(f"State dimension      : {state_dim}")
    print(f"Action dimension     : {action_dim}")
    print(f"Total timesteps      : {config.total_timesteps}")
    print(f"Update timestep      : {config.update_timestep}")
    print(f"Log path             : {log_path}")
    print(f"Last checkpoint      : {last_checkpoint_path}")
    print(f"Best checkpoint      : {best_checkpoint_path}")
    print("============================================================================================")

    start_time = datetime.now().replace(microsecond=0)
    best_eval_reward = -np.inf

    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "episode",
                "timestep",
                "episode_reward",
                "episode_length",
                "success",
                "eval_reward",
                "eval_success_rate",
            ]
        )

        time_step = 0
        episode = 0
        print_reward_acc = 0.0
        print_episode_acc = 0

        try:
            while time_step < config.total_timesteps:
                state, _ = env.reset(seed=config.seed + episode)
                episode_reward = 0.0
                episode_success = False

                for t in range(1, config.max_ep_len + 1):
                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    agent.buffer.rewards.append(reward)
                    agent.buffer.is_terminals.append(done)

                    state = next_state
                    episode_reward += reward
                    episode_success = episode_success or bool(info["is_success"])
                    time_step += 1

                    if time_step % config.update_timestep == 0:
                        agent.update()

                    if time_step % config.action_std_decay_freq == 0:
                        agent.decay_action_std(config.action_std_decay_rate, config.min_action_std)

                    eval_reward = ""
                    eval_success_rate = ""

                    if time_step % config.eval_freq == 0:
                        eval_reward_value, eval_success_value = evaluate_policy(agent, config)
                        eval_reward = f"{eval_reward_value:.4f}"
                        eval_success_rate = f"{eval_success_value:.4f}"

                        if eval_reward_value > best_eval_reward:
                            best_eval_reward = eval_reward_value
                            agent.save(str(best_checkpoint_path))
                            print(
                                f"[best] timestep={time_step} eval_reward={eval_reward_value:.3f} "
                                f"success_rate={eval_success_value:.2%}"
                            )

                    if time_step % config.log_freq == 0 or done:
                        writer.writerow(
                            [
                                episode,
                                time_step,
                                round(episode_reward, 4),
                                t,
                                int(episode_success),
                                eval_reward,
                                eval_success_rate,
                            ]
                        )
                        log_file.flush()

                    if time_step % config.print_freq == 0:
                        avg_reward = print_reward_acc / max(print_episode_acc, 1)
                        print(
                            f"Episode={episode} Timestep={time_step} "
                            f"AvgReward={avg_reward:.3f} CurrentReward={episode_reward:.3f}"
                        )
                        print_reward_acc = 0.0
                        print_episode_acc = 0

                    if time_step % config.save_freq == 0:
                        agent.save(str(last_checkpoint_path))
                        print(f"[save] timestep={time_step} checkpoint={last_checkpoint_path}")

                    if done or time_step >= config.total_timesteps:
                        break

                print_reward_acc += episode_reward
                print_episode_acc += 1
                episode += 1
        finally:
            env.close()

    if agent.buffer.rewards:
        agent.update()

    agent.save(str(last_checkpoint_path))

    metadata = {
        "env_name": env_id,
        "control_space": config.env,
        "started_at": str(start_time),
        "finished_at": str(datetime.now().replace(microsecond=0)),
        "config": asdict(config),
        "last_checkpoint": str(last_checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "best_eval_reward": None if best_eval_reward == -np.inf else best_eval_reward,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("============================================================================================")
    print(f"Finished training in {datetime.now().replace(microsecond=0) - start_time}")
    print(f"Saved last checkpoint to {last_checkpoint_path}")
    if best_eval_reward > -np.inf:
        print(f"Saved best checkpoint to {best_checkpoint_path} (eval_reward={best_eval_reward:.3f})")
    print(f"Saved metadata to {metadata_path}")
    print("============================================================================================")


if __name__ == "__main__":
    train()
