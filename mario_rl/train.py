"""Small-memory PPO training on CPU or Apple Silicon MPS.

Rollouts stay in host memory; only a minibatch is copied to the accelerator.
Each environment step is one four-frame action. A resumed run restores the
optimizer and random states, but starts new games (NES snapshots are not saved).
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import shutil
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn


def compute_gae(rewards, values, next_values, terminated, truncated,
                gamma=0.99, gae_lambda=0.95):
    """Bootstrap time limits, but never propagate advantages into a new game.

    All arrays are [time, environment]. ``next_values`` must describe the
    actual post-action observation, before any automatic reset.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    if not (rewards.shape == values.shape == next_values.shape ==
            terminated.shape == truncated.shape) or rewards.ndim != 2:
        raise ValueError("GAE inputs must have the same [time, environment] shape")
    advantages = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(rewards.shape[0])):
        bootstrap = (~terminated[t]).astype(np.float32)
        continues = (~(terminated[t] | truncated[t])).astype(np.float32)
        delta = rewards[t] + gamma * next_values[t] * bootstrap - values[t]
        carry = delta + gamma * gae_lambda * continues * carry
        advantages[t] = carry
    return advantages, advantages + values


def clipped_policy_loss(new_log_probs, old_log_probs, advantages, clip_coef=0.2):
    """PPO's pessimistic clipped surrogate, returned as a minimization loss."""
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    return -torch.minimum(unclipped, clipped).mean()


def critic_value_loss(values, returns, loss_type="mse"):
    """Keep the legacy half-MSE, or bound residual gradients with beta=1 Huber."""
    if loss_type == "mse":
        return 0.5 * (values - returns).square().mean()
    if loss_type == "huber":
        return nn.functional.smooth_l1_loss(values, returns, beta=1.0)
    raise ValueError(f"Unknown value loss: {loss_type!r}")


def capture_rng_state():
    numpy_state = np.random.get_state()
    result = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.tensor(numpy_state[1].astype(np.int64)),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        numpy_state["keys"].cpu().numpy().astype(np.uint32),
        numpy_state["pos"], numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, global_step, update, config,
                    training_state=None, lineage=None):
    """Atomic checkpoint containing only safe torch.load(weights_only=True) types."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "update": int(update),
        "config": config,
        "rng": capture_rng_state(),
        "resume_semantics": "optimizer_and_rng_restored; environments_reset",
        "training_state": training_state or {},
        "lineage": lineage or {},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def create_optimizer(model, learning_rate=None, optimizer_state=None, scratch=False):
    """Preserve a checkpoint's LR unless the user explicitly overrides it."""
    initial_lr = learning_rate if learning_rate is not None else (2.5e-4 if scratch else 2.5e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, eps=1e-5)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        if learning_rate is not None:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
    return optimizer


def json_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_digest(model):
    """Hash tensor names, shapes, dtypes and bytes independently of torch ZIP metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def source_snapshot(output):
    """Keep the actual source text as well as its content hashes; no git mutation."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "mario_rl").rglob("*.py"))
    paths += sorted(root.glob("requirements*.txt"))
    if (root / "pyproject.toml").exists():
        paths.append(root / "pyproject.toml")
    files = {str(path.relative_to(root)): {"sha256": file_digest(path),
                                          "text": path.read_text(encoding="utf-8")}
             for path in paths}
    digest = json_digest({name: item["sha256"] for name, item in files.items()})
    path = Path(output) / f"source-snapshot-{digest[:16]}.json"
    if not path.exists():
        write_json(path, {"sha256": digest, "files": files})
    return {"sha256": digest, "path": str(path.resolve())}


def prune_checkpoints(output, keep):
    """Bound numbered archives only; never delete initial, best, last or foreign files."""
    archives = []
    for path in Path(output).glob("checkpoint_*.pt"):
        suffix = path.stem.removeprefix("checkpoint_")
        if suffix.isdigit():
            archives.append((int(suffix), path))
    archives.sort()
    for _, path in archives[:max(0, len(archives) - keep)]:
        path.unlink()


def inherit_protocol(args, previous_config=None):
    """A resume preserves reward, termination, and critic-loss contracts."""
    previous_config = previous_config or {}
    for key, new_default, legacy_default in (("reward_mode", "full", "full"),
                                             ("castle_path_rules", False, True),
                                             ("value_loss", "mse", "mse")):
        supplied = getattr(args, key, None)
        if args.resume:
            inherited = previous_config.get(key, legacy_default)
            if supplied is not None and supplied != inherited:
                raise ValueError(f"Resume {key} must match checkpoint ({inherited!r})")
            setattr(args, key, inherited)
        elif supplied is None:
            setattr(args, key, new_default)
    if args.value_loss not in {"mse", "huber"}:
        raise ValueError(f"Unknown value loss: {args.value_loss!r}")


def evaluation_protocol(args, device):
    return {"world": args.world, "stage": args.stage, "reward_mode": args.reward_mode,
            "castle_path_rules": args.castle_path_rules, "episodes": args.eval_episodes,
            "max_steps": args.eval_max_steps, "seed": args.eval_seed,
            "action_selection": "argmax", "fixed_start": True,
            "action_repeat": 4, "device": str(device)}


def evaluation_score(rows):
    if not rows or not all(row.get("complete", False) for row in rows):
        return None
    return (sum(bool(row["flag_get"]) for row in rows) / len(rows),
            float(np.mean([row["raw_return"] for row in rows])),
            -float(np.mean([row["steps"] for row in rows])))


def update_evaluation_state(state, rows, protocol_hash, global_step, model_hash):
    """Compare only complete, periodic evaluations with an identical protocol."""
    score = evaluation_score(rows)
    if score is None:
        return False
    if state.get("evaluation_protocol_sha256") != protocol_hash:
        state["best_evaluation"] = None
        state["consecutive_clear_evals"] = 0
    state["evaluation_protocol_sha256"] = protocol_hash
    state["periodic_evaluations"] = int(state.get("periodic_evaluations", 0)) + 1
    state["consecutive_clear_evals"] = (
        int(state.get("consecutive_clear_evals", 0)) + 1 if score[0] == 1.0 else 0)
    best = state.get("best_evaluation")
    improved = best is None or score > tuple(best["score"])
    if improved:
        state["best_evaluation"] = {
            "global_step": int(global_step), "score": list(score),
            "wins": sum(bool(row["flag_get"]) for row in rows), "episodes": len(rows),
            "model_sha256": model_hash, "protocol_sha256": protocol_hash,
            "periodic_evaluation": state["periodic_evaluations"],
        }
    return improved


def _components(info, key):
    values = info.get(key, {})
    if not isinstance(values, dict):
        raise ValueError(f"Environment {key} must be a dictionary")
    result = {str(name): float(value) for name, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"Non-finite environment {key}")
    return result


def _add_components(total, addition):
    for name, value in addition.items():
        total[name] = total.get(name, 0.0) + value


def evaluate_training_policy(model, env, device, episodes, max_steps, seed, stop_check):
    """Use a separate game and preserve all training RNG states around evaluation."""
    rng_state = capture_rng_state()
    was_training = model.training
    rows = []
    aborted_reason = None
    model.eval()
    try:
        for episode in range(episodes):
            if (aborted_reason := stop_check()):
                break
            obs, info = env.reset(seed=seed + episode)
            shaped_return = raw_return = 0.0
            max_x = int(info.get("x_pos", 0))
            total_components, applied_components = {}, {}
            terminated = truncated = False
            steps = 0
            for _ in range(max_steps):
                if (aborted_reason := stop_check()):
                    break
                with torch.inference_mode():
                    logits, value = model(torch.as_tensor(obs, device=device).unsqueeze(0))
                    if not (torch.isfinite(logits).all() and torch.isfinite(value).all()):
                        raise FloatingPointError("Non-finite evaluation model outputs")
                    action = int(logits.argmax(dim=-1).item())
                obs, reward, terminated, truncated, info = env.step(action)
                raw = float(info.get("raw_reward", reward))
                if not (math.isfinite(float(reward)) and math.isfinite(raw) and np.isfinite(obs).all()):
                    raise FloatingPointError("Non-finite evaluation observation or reward")
                shaped_return += float(reward)
                raw_return += raw
                max_x = max(max_x, int(info.get("x_pos", 0)))
                steps += 1
                _add_components(total_components, _components(info, "reward_components"))
                _add_components(applied_components, _components(info, "applied_reward_components"))
                if terminated or truncated:
                    break
            flag_get = bool(info.get("flag_get", False))
            reason = (aborted_reason or ("level_clear" if flag_get else
                      "environment_truncated" if truncated else
                      "game_timer_expired" if terminated and info.get("time") == 0 else
                      "environment_terminal_without_clear" if terminated else "evaluation_step_limit"))
            rows.append({"episode": episode + 1, "seed": seed + episode,
                         "flag_get": flag_get, "x": int(info.get("x_pos", 0)), "max_x": max_x,
                         "steps": steps, "return": shaped_return, "raw_return": raw_return,
                         "terminated": bool(terminated), "truncated": bool(truncated),
                         "evaluation_limit": not (terminated or truncated or aborted_reason),
                         "termination_reason": reason, "complete": not bool(aborted_reason),
                         "reward_components": json.dumps(total_components, sort_keys=True),
                         "applied_reward_components": json.dumps(applied_components, sort_keys=True)})
            if aborted_reason:
                break
    finally:
        model.train(was_training)
        restore_rng_state(rng_state)
    return rows, aborted_reason


def add_training_arguments(parser):
    parser.add_argument("--steps", type=int, default=32768,
                        help="new environment decisions in this run; divisible by num-envs")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None,
                        help="default: 2.5e-5 fine-tuning, 2.5e-4 scratch; resume preserves saved LR")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--init", default="scratch",
                        help="scratch (default), local (bundled self-trained model), or a weights/checkpoint path")
    parser.add_argument("--resume", type=Path,
                        help="restore model/optimizer/progress/RNG; games restart")
    parser.add_argument("--output", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=8192,
                        help="environment decisions between archived checkpoints")
    parser.add_argument("--keep-checkpoints", type=int, default=3,
                        help="retain this many numbered archives; initial/best/last are always retained")
    parser.add_argument("--reward-mode", choices=["native", "progress_terminal", "progress_score", "full"],
                        default=None, help="new run: full; resume: inherit the saved reward contract")
    parser.add_argument("--castle-path-rules", action=argparse.BooleanOptionalAction, default=None,
                        help="new run: off; resume: inherit (legacy checkpoints used on)")
    parser.add_argument("--env-backend", choices=["serial", "process"], default="serial")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="periodic argmax evaluation interval in collected decisions; 0 disables")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-max-steps", type=int, default=3000)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--eval-device", choices=["current", "cpu"], default="current")
    parser.add_argument("--stop-after-clear-evals", type=int, default=0,
                        help="stop after this many consecutive all-clear periodic evaluations; 0 disables")
    parser.add_argument("--max-hours", type=float,
                        help="wall-clock budget for this invocation, including initial/periodic evaluation")
    parser.add_argument("--stop-file", type=Path,
                        help="save and exit when this file exists; it is never removed automatically")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--value-loss", choices=["mse", "huber"], default=None,
                        help="new run: mse; resume: inherit saved loss (legacy: mse); Huber beta=1")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03,
                        help="stop PPO epochs early above this KL; 0 disables")
    parser.add_argument("--threads", type=int, default=1)
    return parser


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    return add_training_arguments(parser)


def _validate_args(args):
    for key in ("steps", "num_envs", "rollout_steps", "epochs", "minibatch_size",
                "checkpoint_every", "threads", "eval_episodes", "eval_max_steps"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key.replace('_', '-')} must be positive")
    if args.steps % args.num_envs:
        raise ValueError("--steps must be divisible by --num-envs")
    if (args.lr is not None and args.lr <= 0) or args.max_grad_norm <= 0:
        raise ValueError("lr and max-grad-norm must be positive")
    if not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        raise ValueError("gamma and gae-lambda must be in [0, 1]")
    if not 0 < args.clip_coef < 1:
        raise ValueError("clip-coef must be in (0, 1)")
    if args.entropy_coef < 0 or args.value_coef < 0 or args.target_kl < 0:
        raise ValueError("loss coefficients and target-kl must be nonnegative")
    if args.value_loss not in {None, "mse", "huber"}:
        raise ValueError("value-loss must be mse or huber")
    for key in ("eval_every", "stop_after_clear_evals", "keep_checkpoints"):
        if getattr(args, key) < 0:
            raise ValueError(f"{key.replace('_', '-')} must be nonnegative")
    if args.stop_after_clear_evals and not args.eval_every:
        raise ValueError("--stop-after-clear-evals requires --eval-every > 0")
    if args.max_hours is not None and (not math.isfinite(args.max_hours) or args.max_hours <= 0):
        raise ValueError("--max-hours must be finite and positive")


def _open_csv(path, fieldnames):
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8") as previous:
            reader = csv.DictReader(previous)
            old_fields = reader.fieldnames or []
            combined = old_fields + [key for key in fieldnames if key not in old_fields]
            if combined != old_fields:
                temporary = path.with_name(path.name + ".tmp")
                with temporary.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=combined)
                    writer.writeheader()
                    for row in reader:
                        writer.writerow(row)
                temporary.replace(path)
            fieldnames = combined
    stream = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
    return stream, writer


def train(args):
    """Run auditable PPO with bounded checkpoints and interruptible evaluation."""
    invocation_start = time.perf_counter()
    from .env import make_env
    from .model import ActorCritic, load_model, select_device
    from .weights import ensure_weights
    from .vector_env import VectorEnv

    defaults = vars(build_parser().parse_args([]))
    defaults.update(vars(args))
    args = argparse.Namespace(**defaults)
    _validate_args(args)
    args.output = Path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((args.output / name).exists() for name in ("last.pt", "initial.pt")):
        raise ValueError(f"{args.output} already has checkpoints; choose another --output or --resume")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    eval_device = torch.device("cpu") if args.eval_device == "cpu" else device
    checkpoint = None
    parent_checkpoint = None
    source = None
    if args.resume:
        parent_checkpoint = {"path": str(Path(args.resume).resolve()), "sha256": file_digest(args.resume)}
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        for key in ("model", "optimizer", "global_step", "update", "rng"):
            if key not in checkpoint:
                raise ValueError(f"Resume checkpoint is missing {key}")
        previous_config = checkpoint.get("config", {})
        if (previous_config.get("world", args.world) != args.world or
                previous_config.get("stage", args.stage) != args.stage):
            raise ValueError("Resume world/stage must match the checkpoint")
        inherit_protocol(args, previous_config)
        model = load_model(args.resume, device=device)
    else:
        inherit_protocol(args)
        if args.init == "scratch":
            model = ActorCritic().to(device)
        else:
            if args.init == "pretrained":
                raise ValueError("Upstream initialization is not provided; use scratch, local, or an explicit path")
            source = ensure_weights(args.world, args.stage) if args.init == "local" else Path(args.init)
            model = load_model(source, device=device)
    model.train()
    optimizer = create_optimizer(model, args.lr,
                                 checkpoint["optimizer"] if checkpoint else None,
                                 scratch=args.init == "scratch")
    args.lr = float(optimizer.param_groups[0]["lr"])
    global_step = int(checkpoint["global_step"]) if checkpoint else 0
    update = int(checkpoint["update"]) if checkpoint else 0
    initial_step = global_step
    target_step = initial_step + args.steps
    segment_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    starting_model_hash = model_digest(model)
    snapshot = source_snapshot(args.output)
    lineage = copy.deepcopy((checkpoint or {}).get("lineage") or {})
    lineage_path = args.output / "lineage.json"
    if checkpoint and lineage_path.exists():
        existing_lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        if existing_lineage.get("root_initial_model_sha256") == lineage.get("root_initial_model_sha256"):
            lineage = existing_lineage
    if not lineage:
        lineage = {"format_version": 1, "segments": [],
                   "root_initial_model_sha256": starting_model_hash,
                   "initialization": {
                       "kind": "legacy_resume_unknown_origin" if checkpoint else
                               "scratch" if source is None else
                               "bundled_local_model" if args.init == "local" else "local_checkpoint",
                       "seed": args.seed, "source_path": str(Path(source).resolve()) if source else None,
                       "source_sha256": file_digest(source) if source else None,
                       "legacy_parent": parent_checkpoint if checkpoint else None,
                       "origin_is_known": checkpoint is None,
                   }}
    lineage.setdefault("segments", [])
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items() if not callable(value)}
    config.update({"resolved_device": str(device), "resolved_eval_device": str(eval_device),
                   "initial_step": initial_step, "target_step": target_step,
                   "environment_steps_per_decision": 4, "resume_restarts_games": True,
                   "segment_id": segment_id, "initialization_provenance": lineage["initialization"],
                   "source_snapshot_sha256": snapshot["sha256"],
                   "starting_model_sha256": starting_model_hash})
    config["config_sha256"] = json_digest(config)
    config_path = args.output / (f"config-resume-{initial_step}-{segment_id}.json" if checkpoint else "config.json")
    write_json(config_path, config)
    segment = {"segment_id": segment_id, "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "parent_checkpoint": parent_checkpoint, "starting_global_step": initial_step,
               "starting_model_sha256": starting_model_hash, "config_sha256": config["config_sha256"],
               "config_path": str(config_path.resolve()), "source_snapshot": snapshot,
               "requested_init_ignored_on_resume": args.init if checkpoint else None}
    lineage["segments"].append(segment)
    protocol = evaluation_protocol(args, eval_device)
    protocol_hash = json_digest(protocol)
    state = copy.deepcopy((checkpoint or {}).get("training_state") or {})
    state.setdefault("periodic_evaluations", 0)
    state.setdefault("consecutive_clear_evals", 0)
    state.setdefault("best_evaluation", None)
    state.setdefault("total_completed_episodes", 0)
    state.setdefault("optimizer_steps", 0)
    state.setdefault("optimization_minibatch_samples", 0)
    state.setdefault("discarded_rollout_decisions", 0)
    if state.get("evaluation_protocol_sha256") not in (None, protocol_hash):
        state["previous_evaluation_protocol_sha256"] = state.get("evaluation_protocol_sha256")
        state["consecutive_clear_evals"] = 0
        state["best_evaluation"] = None
        state["next_evaluation_step"] = global_step + args.eval_every
        state["evaluation_history_note"] = "Protocol changed: best comparison and consecutive count restarted."
    elif checkpoint and not checkpoint.get("training_state"):
        state["evaluation_history_note"] = "Legacy checkpoint has no evaluation counts; starting at zero."
    state["evaluation_protocol_sha256"] = protocol_hash
    state["evaluation_protocol"] = protocol
    state.setdefault("next_evaluation_step", global_step + args.eval_every)
    if state["next_evaluation_step"] <= global_step:
        state["next_evaluation_step"] = global_step + args.eval_every
    state["segment_id"] = segment_id
    state["stop_reason"] = "running"
    state["evaluation_enabled"] = bool(args.eval_every)

    initial_path = args.output / "initial.pt"
    if not initial_path.exists():
        old_initial = Path(lineage.get("initial_checkpoint_path", ""))
        expected_digest = lineage.get("initial_checkpoint_sha256")
        if checkpoint and old_initial.is_file() and expected_digest == file_digest(old_initial):
            shutil.copy2(old_initial, initial_path)
        else:
            lineage["initial_model_sha256"] = starting_model_hash
            lineage["initial_baseline_scope"] = "resume_segment_only" if checkpoint else "root_initialization"
            save_checkpoint(initial_path, model, optimizer, global_step, update, config, state, lineage)
    elif lineage.get("initial_checkpoint_sha256") and file_digest(initial_path) != lineage["initial_checkpoint_sha256"]:
        raise ValueError("Existing initial.pt does not match the preserved lineage")
    lineage["initial_checkpoint_path"] = str(initial_path.resolve())
    lineage["initial_checkpoint_sha256"] = file_digest(initial_path)
    lineage.setdefault("initial_model_sha256", lineage["root_initial_model_sha256"])
    write_json(lineage_path, lineage)
    if state.get("best_evaluation") and not (args.output / "best.pt").exists():
        previous_best = Path(state.get("best_checkpoint_path", ""))
        if previous_best.is_file() and state.get("best_checkpoint_sha256") == file_digest(previous_best):
            shutil.copy2(previous_best, args.output / "best.pt")
            state["best_checkpoint_path"] = str((args.output / "best.pt").resolve())
        else:
            state["best_evaluation"] = None
            state["best_restore_note"] = "Prior best checkpoint unavailable; next complete evaluation establishes a new local best."

    metric_fields = ["global_step", "update", "elapsed_seconds", "decisions_per_second", "episodes",
                     "mean_episode_return", "mean_episode_x", "max_x", "success_rate", "policy_loss",
                     "value_loss", "entropy", "approx_kl", "clip_fraction", "epochs_completed",
                     "mean_raw_episode_return", "segment_id", "partial_update"]
    episode_fields = ["global_step", "environment", "return", "length", "max_x", "flag_get", "terminated",
                      "truncated", "raw_return", "reward_mode", "reward_components", "applied_reward_components",
                      "reward_native", "reward_score", "reward_terminal", "reward_castle_rule",
                      "applied_reward_native", "applied_reward_score", "applied_reward_terminal",
                      "applied_reward_castle_rule", "complete", "termination_reason", "segment_id"]
    evaluation_fields = ["global_step", "evaluation_index", "phase", "model_sha256", "protocol_sha256",
                         "episode", "seed", "flag_get", "x", "max_x", "steps", "return", "raw_return",
                         "terminated", "truncated", "evaluation_limit", "termination_reason", "complete",
                         "reward_components", "applied_reward_components", "segment_id"]
    metrics_file, metrics_writer = _open_csv(args.output / "metrics.csv", metric_fields)
    episodes_file, episodes_writer = _open_csv(args.output / "episodes.csv", episode_fields)
    evaluation_file, evaluation_writer = _open_csv(args.output / "evaluation.csv", evaluation_fields)
    vector = eval_env = None
    recent_episodes = deque(maxlen=100)
    segment_episodes = 0
    maximum_x = 0
    stop_reason = None
    failure = None
    next_checkpoint = global_step + args.checkpoint_every
    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    raw_episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
    episode_max_x = np.zeros(args.num_envs, dtype=np.int64)
    episode_components = [{} for _ in range(args.num_envs)]
    applied_episode_components = [{} for _ in range(args.num_envs)]

    def stop_check():
        if args.stop_file is not None and Path(args.stop_file).exists():
            return "stop_file"
        if args.max_hours is not None and time.perf_counter() - invocation_start >= args.max_hours * 3600:
            return "max_hours"
        return None

    def persist_state(reason="running"):
        state.update({"global_step": global_step, "update": update, "stop_reason": reason,
                      "elapsed_seconds": round(time.perf_counter() - invocation_start, 3),
                      "new_decisions": global_step - initial_step, "segment_episodes": segment_episodes})
        write_json(args.output / "training-state.json", state)

    def checkpoint_to(path):
        persist_state(stop_reason or "running")
        save_checkpoint(path, model, optimizer, global_step, update, config, state, lineage)

    def record_episode(i, info, term=False, trunc=False, complete=True, reason=None):
        nonlocal segment_episodes
        components = episode_components[i]
        applied = applied_episode_components[i]
        row = {"global_step": global_step, "environment": i, "return": float(episode_returns[i]),
               "raw_return": float(raw_episode_returns[i]), "length": int(episode_lengths[i]),
               "max_x": int(episode_max_x[i]), "flag_get": bool(info.get("flag_get", False)),
               "terminated": bool(term), "truncated": bool(trunc), "reward_mode": args.reward_mode,
               "reward_components": json.dumps(components, sort_keys=True),
               "applied_reward_components": json.dumps(applied, sort_keys=True),
               "complete": bool(complete), "segment_id": segment_id,
               "termination_reason": reason or ("level_clear" if info.get("flag_get") else
                   "environment_truncated" if trunc else "game_timer_expired" if info.get("time") == 0
                   else "environment_terminal_without_clear")}
        for key in ("native", "score", "terminal", "castle_rule"):
            row[f"reward_{key}"] = components.get(key, 0.0)
            row[f"applied_reward_{key}"] = applied.get(key, 0.0)
        episodes_writer.writerow(row)
        if complete:
            recent_episodes.append(row)
            segment_episodes += 1
            state["total_completed_episodes"] += 1
        episode_returns[i] = raw_episode_returns[i] = episode_lengths[i] = episode_max_x[i] = 0
        episode_components[i], applied_episode_components[i] = {}, {}

    def run_evaluation(phase):
        nonlocal stop_reason
        evaluated_model = model if eval_device == device else copy.deepcopy(model).to(eval_device)
        model_hash = model_digest(model)
        rows, aborted = evaluate_training_policy(evaluated_model, eval_env, eval_device,
                            args.eval_episodes, args.eval_max_steps, args.eval_seed, stop_check)
        index = state["periodic_evaluations"] + (1 if phase == "periodic" else 0)
        for row in rows:
            evaluation_writer.writerow(dict(row, global_step=global_step, evaluation_index=index,
                phase=phase, model_sha256=model_hash, protocol_sha256=protocol_hash, segment_id=segment_id))
        evaluation_file.flush()
        if aborted:
            stop_reason = aborted
            state["last_evaluation_incomplete"] = True
            return
        if len(rows) != args.eval_episodes:
            raise RuntimeError("Evaluation returned fewer episodes without a stop reason")
        state["last_evaluation_incomplete"] = False
        score = evaluation_score(rows)
        print(f"evaluation phase={phase} step={global_step} wins={sum(r['flag_get'] for r in rows)}/{len(rows)} "
              f"mean_raw_return={score[1]:.3f}", flush=True)
        if phase == "initial":
            state["initial_evaluation"] = {"global_step": global_step, "score": list(score),
                "model_sha256": model_hash, "protocol_sha256": protocol_hash, "episodes": len(rows)}
        else:
            improved = update_evaluation_state(state, rows, protocol_hash, global_step, model_hash)
            state["next_evaluation_step"] = global_step + args.eval_every
            if improved:
                state["best_checkpoint_path"] = str((args.output / "best.pt").resolve())
                checkpoint_to(args.output / "best.pt")
                state["best_checkpoint_sha256"] = file_digest(args.output / "best.pt")
            if args.stop_after_clear_evals and state["consecutive_clear_evals"] >= args.stop_after_clear_evals:
                stop_reason = "clear_evals"
        persist_state(stop_reason or "running")

    estimated_mib = args.rollout_steps * args.num_envs * 4 * 84 * 84 * 4 / 2**20
    print(f"PPO device={device} | backend={args.env_backend} | environments={args.num_envs} | "
          f"reward={args.reward_mode} | new decisions={args.steps} | buffer={estimated_mib:.1f} MiB", flush=True)
    try:
        stop_reason = stop_check()
        if not stop_reason:
            vector = VectorEnv(num_envs=args.num_envs, world=args.world, stage=args.stage, seed=args.seed,
                               reward_mode=args.reward_mode, castle_path_rules=args.castle_path_rules,
                               backend=args.env_backend)
            observations, _ = vector.reset()
            if checkpoint:
                restore_rng_state(checkpoint["rng"])
            if args.eval_every:
                eval_env = make_env(world=args.world, stage=args.stage, seed=args.eval_seed,
                    reward_mode=args.reward_mode, castle_path_rules=args.castle_path_rules,
                    max_steps=max(5000, args.eval_max_steps + 1))
                if not checkpoint:
                    run_evaluation("initial")
            persist_state(stop_reason or "running")
        while not stop_reason and global_step < target_step:
            if (stop_reason := stop_check()):
                break
            rollout_size = min(args.rollout_steps, (target_step - global_step) // args.num_envs)
            shape = (rollout_size, args.num_envs)
            states = np.empty(shape + tuple(observations.shape[1:]), dtype=np.float32)
            actions = np.empty(shape, dtype=np.int64)
            old_log_probs = np.empty(shape, dtype=np.float32)
            rewards = np.empty(shape, dtype=np.float32)
            values = np.empty(shape, dtype=np.float32)
            next_values = np.empty(shape, dtype=np.float32)
            terminated = np.empty(shape, dtype=bool)
            truncated = np.empty(shape, dtype=bool)
            collected = 0
            for t in range(rollout_size):
                if (stop_reason := stop_check()):
                    break
                if not np.isfinite(observations).all():
                    raise FloatingPointError("Non-finite training observations")
                states[t] = observations
                with torch.no_grad():
                    logits, value = model(torch.as_tensor(observations, device=device))
                    if not (torch.isfinite(logits).all() and torch.isfinite(value).all()):
                        raise FloatingPointError("Non-finite rollout model outputs")
                    log_probs_cpu = logits.detach().cpu().log_softmax(dim=-1)
                    chosen = torch.multinomial(log_probs_cpu.exp(), 1).squeeze(1)
                    actions[t] = chosen.numpy()
                    old_log_probs[t] = log_probs_cpu.gather(1, chosen[:, None]).squeeze(1).numpy()
                    values[t] = value.squeeze(-1).detach().cpu().numpy()
                post_action, step_rewards, terms, truncs, infos = vector.step(actions[t])
                if not (np.isfinite(post_action).all() and np.isfinite(step_rewards).all()):
                    raise FloatingPointError("Non-finite vector environment observations or rewards")
                rewards[t], terminated[t], truncated[t] = step_rewards, terms, truncs
                global_step += args.num_envs
                collected += args.num_envs
                for i, info in enumerate(infos):
                    raw = float(info.get("raw_reward", step_rewards[i]))
                    if not math.isfinite(raw):
                        raise FloatingPointError("Non-finite raw reward")
                    episode_returns[i] += float(step_rewards[i])
                    raw_episode_returns[i] += raw
                    episode_lengths[i] += 1
                    episode_max_x[i] = max(episode_max_x[i], int(info.get("x_pos", 0)))
                    maximum_x = max(maximum_x, int(episode_max_x[i]))
                    _add_components(episode_components[i], _components(info, "reward_components"))
                    _add_components(applied_episode_components[i], _components(info, "applied_reward_components"))
                    if terms[i] or truncs[i]:
                        record_episode(i, info, bool(terms[i]), bool(truncs[i]))
                # Bootstrap the final observation before resetting any game.
                with torch.no_grad():
                    _, next_value = model(torch.as_tensor(post_action, device=device))
                    if not torch.isfinite(next_value).all():
                        raise FloatingPointError("Non-finite bootstrap values")
                    next_values[t] = next_value.squeeze(-1).detach().cpu().numpy()
                observations = post_action
                ended = np.flatnonzero(terms | truncs).tolist()
                if ended:
                    for i, (obs, _) in vector.reset_at(ended).items():
                        observations[i] = obs
            if stop_reason:
                state["discarded_rollout_decisions"] += collected
                break
            advantages, returns = compute_gae(rewards, values, next_values, terminated, truncated,
                                               args.gamma, args.gae_lambda)
            flat_states = states.reshape((-1,) + tuple(states.shape[2:]))
            flat_actions = actions.reshape(-1)
            flat_old_log_probs = old_log_probs.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
            flat_returns = returns.reshape(-1)
            minibatch_metrics = []
            epochs_completed = 0
            for epoch in range(args.epochs):
                indices = np.random.permutation(len(flat_actions))
                epoch_kls = []
                for offset in range(0, len(indices), args.minibatch_size):
                    if (stop_reason := stop_check()):
                        break
                    batch = indices[offset:offset + args.minibatch_size]
                    batch_states = torch.as_tensor(flat_states[batch], device=device)
                    batch_actions = torch.as_tensor(flat_actions[batch], device=device)
                    batch_old = torch.as_tensor(flat_old_log_probs[batch], device=device)
                    batch_adv = torch.as_tensor(flat_advantages[batch], device=device)
                    batch_returns = torch.as_tensor(flat_returns[batch], device=device)
                    logits, new_values = model(batch_states)
                    all_log_probs = logits.log_softmax(dim=-1)
                    new_log_probs = all_log_probs.gather(1, batch_actions[:, None]).squeeze(1)
                    policy_loss = clipped_policy_loss(new_log_probs, batch_old, batch_adv, args.clip_coef)
                    value_loss = critic_value_loss(new_values.squeeze(-1), batch_returns,
                                                   args.value_loss)
                    entropy = -(all_log_probs.exp() * all_log_probs).sum(-1).mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("Non-finite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    state["optimizer_steps"] += 1
                    state["optimization_minibatch_samples"] += len(batch)
                    with torch.no_grad():
                        log_ratio = new_log_probs - batch_old
                        ratio = log_ratio.exp()
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        clip_fraction = ((ratio - 1).abs() > args.clip_coef).float().mean().item()
                    epoch_kls.append(approx_kl)
                    minibatch_metrics.append([policy_loss.item(), value_loss.item(), entropy.item(),
                                              approx_kl, clip_fraction])
                if stop_reason:
                    break
                epochs_completed += 1
                if args.target_kl and np.mean(epoch_kls) > args.target_kl:
                    break
            if minibatch_metrics:
                update += 1
                elapsed = time.perf_counter() - invocation_start
                row = dict(zip(["policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"],
                               map(float, np.mean(minibatch_metrics, axis=0))))
                row.update({"global_step": global_step, "update": update, "elapsed_seconds": round(elapsed, 3),
                    "decisions_per_second": round((global_step - initial_step) / max(elapsed, 1e-6), 2),
                    "episodes": segment_episodes, "max_x": maximum_x, "segment_id": segment_id,
                    "mean_episode_return": float(np.mean([r["return"] for r in recent_episodes])) if recent_episodes else None,
                    "mean_raw_episode_return": float(np.mean([r["raw_return"] for r in recent_episodes])) if recent_episodes else None,
                    "mean_episode_x": float(np.mean([r["max_x"] for r in recent_episodes])) if recent_episodes else None,
                    "success_rate": float(np.mean([r["flag_get"] for r in recent_episodes])) if recent_episodes else None,
                    "epochs_completed": epochs_completed, "partial_update": bool(stop_reason)})
                metrics_writer.writerow(row)
                metrics_file.flush()
                episodes_file.flush()
                print(f"step={global_step}/{target_step} update={update} speed={row['decisions_per_second']:.1f} "
                      f"decisions/s episodes={segment_episodes} max_x={maximum_x} entropy={row['entropy']:.4f}", flush=True)
            if not stop_reason and args.eval_every and global_step >= state["next_evaluation_step"]:
                run_evaluation("periodic")
            persist_state(stop_reason or "running")
            if global_step >= next_checkpoint:
                checkpoint_to(args.output / f"checkpoint_{global_step}.pt")
                checkpoint_to(args.output / "last.pt")
                prune_checkpoints(args.output, args.keep_checkpoints)
                next_checkpoint = global_step + args.checkpoint_every
        stop_reason = stop_reason or "steps_completed"
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("Interrupted; saving current model and optimizer.", flush=True)
    except Exception as error:
        stop_reason = "error"
        failure = error
        state["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        for i in range(args.num_envs):
            if episode_lengths[i]:
                record_episode(i, {}, complete=False, reason=f"training_{stop_reason or 'stopped'}")
        persist_state(stop_reason or "error")
        segment.update({"ended_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                        "ending_global_step": global_step, "ending_update": update,
                        "ending_model_sha256": model_digest(model), "stop_reason": stop_reason,
                        "elapsed_seconds": round(time.perf_counter() - invocation_start, 3)})
        write_json(lineage_path, lineage)
        checkpoint_to(args.output / "last.pt")
        prune_checkpoints(args.output, args.keep_checkpoints)
        if vector is not None:
            vector.close()
        if eval_env is not None:
            eval_env.close()
        for stream in (metrics_file, episodes_file, evaluation_file):
            stream.close()
    summary = {"device": str(device), "initial_step": initial_step, "global_step": global_step,
               "new_decisions": global_step - initial_step, "updates": update,
               "elapsed_seconds": round(time.perf_counter() - invocation_start, 3),
               "episodes": segment_episodes, "total_completed_episodes": state["total_completed_episodes"],
               "max_x": maximum_x, "stop_reason": stop_reason,
               "interrupted": stop_reason in ("keyboard_interrupt", "max_hours", "stop_file"),
               "checkpoint": str(args.output / "last.pt"), "resumed": checkpoint is not None,
               "resume_restarts_games": True, "reward_mode": args.reward_mode,
               "castle_path_rules": args.castle_path_rules, "initial_model_sha256": lineage["initial_model_sha256"],
               "initialization_provenance": lineage["initialization"], "config_sha256": config["config_sha256"],
               "source_snapshot_sha256": snapshot["sha256"], "segment_id": segment_id,
               "periodic_evaluations": state["periodic_evaluations"],
               "consecutive_clear_evals": state["consecutive_clear_evals"],
               "best_evaluation": state.get("best_evaluation"),
               "optimizer_steps": state["optimizer_steps"],
               "note": "Periodic fixed-start argmax evaluation is separate from stochastic training. Initial baseline does not count toward early stopping."}
    if failure:
        summary["error"] = state["error"]
    write_json(args.output / "summary.json", summary)
    write_json(args.output / f"summary-{segment_id}.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if failure:
        raise failure
    return summary

def main():
    args = build_parser().parse_args()
    try:
        train(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
