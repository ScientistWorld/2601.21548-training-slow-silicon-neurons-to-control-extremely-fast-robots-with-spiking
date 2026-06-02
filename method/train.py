#!/usr/bin/env python3
"""Train the paper's spiking e-prop controller and write evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.air_hockey_env import AirHockeyInterceptionEnv
from method.spiking_eprop_rl import SpikingAgentConfig, SpikingEpropAgent


def run_episode(env: AirHockeyInterceptionEnv, agent: SpikingEpropAgent, train: bool) -> dict:
    state = env.reset()
    total_reward = 0.0
    trajectory = []
    done = False
    info = {"success": False, "hit": False, "steps": 0}
    while not done:
        action, probs, trace = agent.act(state, greedy=not train)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if train:
            trajectory.append({"trace": trace, "probs": probs, "action": action, "reward": reward})
        state = next_state
    if train:
        agent.finish_episode(trajectory)
    return {"success": bool(info["success"]), "hit": bool(info["hit"]), "steps": int(info["steps"]), "reward": float(total_reward)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="random_position_speed")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--hidden-neurons", type=int, default=1020)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--entropy-floor", type=float, default=0.08)
    parser.add_argument("--output", default="outputs/spiking_eprop_random_position_speed.json")
    args = parser.parse_args()

    all_train = []
    all_eval = []
    for seed in range(args.seeds):
        cfg = SpikingAgentConfig(
            hidden_neurons=args.hidden_neurons,
            learning_rate=args.learning_rate,
            entropy_floor=args.entropy_floor,
        )
        agent = SpikingEpropAgent(seed=seed, config=cfg)
        train_env = AirHockeyInterceptionEnv(condition=args.condition, seed=1000 + seed)
        train_records = []
        for _ in range(args.episodes):
            train_records.append(run_episode(train_env, agent, train=True))
        eval_env = AirHockeyInterceptionEnv(condition=args.condition, seed=9000 + seed)
        eval_records = [run_episode(eval_env, agent, train=False) for _ in range(args.eval_episodes)]
        all_train.append(train_records)
        all_eval.append(eval_records)

    eval_success = [float(np.mean([r["success"] for r in records])) for records in all_eval]
    eval_hit = [float(np.mean([r["hit"] for r in records])) for records in all_eval]
    train_tail_success = [
        float(np.mean([r["success"] for r in records[-min(100, len(records)) :]])) for records in all_train
    ]
    train_first_success = [
        float(np.mean([r["success"] for r in records[: min(100, len(records))]])) for records in all_train
    ]
    train_mid_success = [
        float(np.mean([r["success"] for r in records[len(records) // 2 : len(records) // 2 + min(100, len(records))]]))
        for records in all_train
    ]
    result = {
        "method": "spiking_eprop",
        "condition": args.condition,
        "episodes": args.episodes,
        "eval_episodes": args.eval_episodes,
        "seeds": args.seeds,
        "controller_units": args.hidden_neurons,
        "hidden_neurons": args.hidden_neurons,
        "learning_rate": args.learning_rate,
        "entropy_floor": args.entropy_floor,
        "success_rate": float(np.mean(eval_success)),
        "success_rate_iqr": float(np.subtract(*np.percentile(eval_success, [75, 25]))),
        "hit_rate": float(np.mean(eval_hit)),
        "train_tail_success_rate": float(np.mean(train_tail_success)),
        "train_first_success_rate": float(np.mean(train_first_success)),
        "train_mid_success_rate": float(np.mean(train_mid_success)),
        "eval_success_by_seed": eval_success,
        "train_tail_success_by_seed": train_tail_success,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
