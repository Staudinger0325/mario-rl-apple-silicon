"""Command line entry points: doctor, play, human, evaluate, train."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def doctor(args):
    import importlib.metadata
    import numpy as np
    import torch
    from .env import make_env
    from .model import ActorCritic
    data = dict(platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
                mps_built=torch.backends.mps.is_built(), mps_available=torch.backends.mps.is_available(),
                packages={k: importlib.metadata.version(k) for k in ["torch", "numpy", "gym", "gym-super-mario-bros", "nes-py", "pygame"]})
    env = make_env()
    start = time.perf_counter()
    try:
        obs, info = env.reset(seed=0)
        max_x = 0
        for i in range(450):
            obs, _, terminated, truncated, info = env.step(4 if i % 30 < 20 else 1)
            max_x = max(max_x, int(info["x_pos"]))
            if terminated or truncated:
                obs, info = env.reset()
        data["game"] = dict(decisions=450, elapsed_seconds=round(time.perf_counter() - start, 3),
                            max_x=max_x, shape=list(obs.shape), dtype=str(obs.dtype), finite=bool(np.isfinite(obs).all()))
        if args.window:
            from .viewer import Viewer
            view = Viewer(fps=30)
            try:
                for i in range(90):
                    if not view.poll():
                        break
                    obs, reward, terminated, truncated, info = env.step(4)
                    view.draw(env.render(), info, policy="SMOKE TEST", step=i, episode=1, reward=reward)
                    if terminated or truncated:
                        obs, info = env.reset()
            finally:
                view.close()
            data["window_test"] = "passed"
    finally:
        env.close()
    if data["mps_available"]:
        model = ActorCritic().to("mps")
        logits, value = model(torch.zeros(1, 4, 84, 84, device="mps"))
        (logits.mean() + value.mean()).backward()
        torch.mps.synchronize()
        data["mps_forward_backward"] = "passed"
    print(json.dumps(data, indent=2))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def play(args):
    import numpy as np
    import torch
    from .model import select_device
    from .evaluate import policy_model, choose_action
    from .viewer import Viewer
    human = args.command == "human"
    if args.policy == "random" and args.checkpoint:
        raise ValueError("--checkpoint cannot be combined with --policy random")
    torch.set_num_threads(1)
    device = select_device(args.device)
    rng = np.random.default_rng(args.seed)
    if human:
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        env = JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{args.world}-{args.stage}-v0"), SIMPLE_MOVEMENT)
        obs, info = env.reset(), {}
        model = None
    else:
        from .env import make_env
        model = policy_model(args.policy, args.checkpoint, device, args.world, args.stage, args.manifest)
        env = make_env(world=args.world, stage=args.stage, seed=args.seed, reward_mode="full", castle_path_rules=False)
        obs, info = env.reset(seed=args.seed)
    view = Viewer(fps=60 if human else args.fps)
    policy_label = ("HUMAN" if human else "CHECKPOINT" if args.checkpoint else "RANDOM" if args.policy == "random"
                    else "OWN-TRAJECTORY BC" if (args.world, args.stage) == (1, 3) else "LOCAL PPO")
    total = 0.0
    episode = 1
    step = 0
    reset_at = None
    try:
        while view.poll():
            if view.reset_requested or (reset_at is not None and time.monotonic() >= reset_at):
                if human:
                    obs, info = env.reset(), {}
                else:
                    obs, info = env.reset()
                total = 0.0
                episode += 1
                reset_at = None
            if reset_at is not None:
                view.draw(obs if human else env.render(), info, policy=policy_label,
                          step=step, episode=episode, reward=total, device=str(device))
                continue
            if not view.paused:
                action = view.human_action() if human else choose_action(model, obs, rng, device)
                if human:
                    obs, reward, terminated, info = env.step(action)
                    truncated = False
                else:
                    obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                step += 1
                frame = obs if human else env.render()
                view.draw(frame, info, policy=policy_label, step=step,
                          episode=episode, reward=total, device=str(device))
                if args.screenshot and step == 60:
                    import pygame
                    path = Path(args.screenshot)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pygame.image.save(view.screen, str(path))
                if terminated or truncated:
                    print(json.dumps(dict(episode=episode, flag_get=bool(info.get("flag_get")), score=int(info.get("score", 0)), return_=total)), flush=True)
                    reset_at = time.monotonic() + (2.0 if info.get("flag_get") else 0.7)
            else:
                view.draw(obs if human else env.render(), info, policy=policy_label,
                          step=step, episode=episode, reward=total, device=str(device))
            if args.max_steps and step >= args.max_steps:
                break
    finally:
        env.close()
        view.close()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        sys.argv.pop(1)
        from .train import main as train_main
        return train_main()
    parser = argparse.ArgumentParser(description="Locally trained Mario policies for Apple Silicon")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Train PPO on CPU/MPS; use train --help for options")
    p = sub.add_parser("doctor", help="Test emulator, native window and Apple GPU")
    p.add_argument("--window", action="store_true")
    p.add_argument("--output", default="outputs/doctor.json")
    p.set_defaults(func=doctor)
    for name in ["play", "human", "evaluate"]:
        p = sub.add_parser(name)
        p.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--world", type=int, choices=range(1, 9), default=1, help="Target world (1-8)")
        p.add_argument("--stage", type=int, choices=range(1, 5), default=1, help="Target stage (1-4)")
        p.add_argument("--policy", choices=["local", "random"], default="local", help="Use the bundled self-trained policy for the requested World 1 level, or random actions")
        p.add_argument("--checkpoint", help="Use a saved local training checkpoint")
        p.add_argument("--manifest", help="Optional local release manifest; no network download")
        if name == "evaluate":
            p.add_argument("--episodes", type=positive, default=5)
            p.add_argument("--max-steps", type=positive, default=3000)
            p.add_argument("--output", default="outputs/evaluation")
            p.add_argument("--record", action="store_true", help="Record first episode to MP4")
            from .evaluate import evaluate
            p.set_defaults(func=evaluate)
        else:
            p.add_argument("--max-steps", type=positive)
            p.add_argument("--fps", type=positive, default=15)
            p.add_argument("--screenshot")
            p.set_defaults(func=play)
    args = parser.parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Stopped.")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
