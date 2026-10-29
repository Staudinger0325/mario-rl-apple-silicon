"""Seeded evaluation and video export with machine-readable results."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from .env import make_env
from .model import load_model, select_device
from .weights import resolve_weights, tensor_digest


def telemetry(info):
    """NES RAM values include NumPy scalars; preserve them as JSON numbers."""
    return {key: value.item() if isinstance(value, np.generic) else value
            for key, value in info.items()}


def policy_checkpoint(policy, checkpoint, world=1, stage=1, manifest=None):
    if policy not in {"local", "random"}:
        raise ValueError("Use --policy local or random; upstream weights are not downloaded")
    if policy == "random":
        if checkpoint:
            raise ValueError("--checkpoint cannot be combined with --policy random")
        return None, None
    if checkpoint:
        return Path(checkpoint).expanduser().resolve(strict=True), None
    return resolve_weights(world, stage, manifest)


def policy_model(policy, checkpoint, device, world=1, stage=1, manifest=None):
    path, record = policy_checkpoint(policy, checkpoint, world, stage, manifest)
    if path is None:
        return None
    model = load_model(path, device=device)
    if record is not None and tensor_digest(model.state_dict()) != record["model_sha256"]:
        raise ValueError("Loaded model differs from the release manifest")
    return model


def choose_action(model, obs, rng, device):
    if model is None:
        return int(rng.integers(7))
    with torch.inference_mode():
        logits, _ = model(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        return int(logits.argmax(dim=-1).item())


def evaluate(args):
    if args.policy == "random" and args.checkpoint:
        raise ValueError("--checkpoint cannot be combined with --policy random")
    device = select_device(args.device)
    world, stage = getattr(args, "world", 1), getattr(args, "stage", 1)
    torch.set_num_threads(1)
    manifest = getattr(args, "manifest", None)
    checkpoint_path, published_record = policy_checkpoint(args.policy, args.checkpoint, world, stage, manifest)
    model = policy_model(args.policy, args.checkpoint, device, world, stage, manifest)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() if checkpoint_path else None
    env = make_env(world=world, stage=stage, seed=args.seed, reward_mode="full", castle_path_rules=False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    episodes = []
    writer = None
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            rng = np.random.default_rng(seed)
            obs, info = env.reset(seed=seed)
            if args.record and episode == 0:
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(output / "episode-1.mp4"), fps=15, codec="libx264", macro_block_size=1)
                writer.append_data(env.render())
            total = 0.0
            raw_total = 0.0
            max_x = int(info.get("x_pos", 0))
            last_best_step = 0
            initial_info = telemetry(info)
            terminated = truncated = False
            start = time.perf_counter()
            for step in range(1, args.max_steps + 1):
                action = choose_action(model, obs, rng, device)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                raw_total += float(info.get("raw_reward", reward))
                current_x = int(info.get("x_pos", 0))
                if current_x > max_x:
                    max_x = current_x
                    last_best_step = step
                if writer is not None:
                    writer.append_data(env.render())
                if terminated or truncated:
                    break
            elapsed = time.perf_counter() - start
            if info.get("flag_get", False):
                reason = "level_clear"
            elif info.get("termination_reason"):
                reason = str(info["termination_reason"])
            elif truncated:
                reason = "environment_time_limit"
            elif terminated:
                reason = "game_timer_expired" if info.get("time") == 0 else "environment_terminal_without_clear"
            else:
                reason = "evaluation_step_limit"
            record = dict(episode=episode + 1, seed=seed, return_=round(total, 4),
                          raw_return=round(raw_total, 4),
                          max_x=max_x, flag_get=bool(info.get("flag_get", False)),
                          score=int(info.get("score", 0)), coins=int(info.get("coins", 0)),
                          steps=step, seconds=round(elapsed, 3),
                          decisions_per_second=round(step / max(elapsed, 1e-9), 2),
                          terminated=bool(terminated), truncated=bool(truncated),
                          evaluation_limit=not (terminated or truncated),
                          termination_reason=reason, steps_since_best_x=step - last_best_step,
                          initial_info=initial_info, final_info=telemetry(info))
            episodes.append(record)
            print(json.dumps(record), flush=True)
            if writer is not None:
                writer.close()
                writer = None
            if episode == 0:
                from PIL import Image
                Image.fromarray(env.render()).save(output / "final-frame.png")
    finally:
        if writer is not None:
            writer.close()
        env.close()
    result = dict(policy="checkpoint" if args.checkpoint else args.policy,
                  method=published_record["method"] if published_record else "custom_checkpoint" if args.checkpoint else "random",
                  checkpoint=str(checkpoint_path) if checkpoint_path else None,
                  checkpoint_sha256=checkpoint_sha256,
                  created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  device=str(device), environment=f"SuperMarioBros-{world}-{stage}-v0",
                  world=world, stage=stage, max_steps=args.max_steps,
                  action_repeat=4, deterministic_policy=args.policy != "random",
                  note="Fixed starting state. Different seeds do not imply different levels or layouts.",
                  episodes=episodes,
                  summary=dict(episodes=len(episodes), wins=sum(x["flag_get"] for x in episodes),
                               win_rate=float(np.mean([x["flag_get"] for x in episodes])),
                               mean_max_x=float(np.mean([x["max_x"] for x in episodes])),
                               mean_return=float(np.mean([x["return_"] for x in episodes]))))
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "episodes.csv").open("w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(episodes[0]))
        writer_csv.writeheader()
        writer_csv.writerows({**row, "initial_info": json.dumps(row["initial_info"]),
                              "final_info": json.dumps(row["final_info"])} for row in episodes)
    print(json.dumps(result["summary"], indent=2))
    return result
