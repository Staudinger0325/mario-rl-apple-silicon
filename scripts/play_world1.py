#!/usr/bin/env python3
"""Run the four accepted, locally trained World 1 policies from fresh starts."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mario_rl.model import load_model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_models(manifest_path):
    """Verify the public state-dict manifest; do not impersonate private receipt audits."""
    from mario_rl.weights import load_manifest, resolve_weights, tensor_digest
    path, manifest = load_manifest(manifest_path)
    models = {}
    for record in manifest["models"]:
        weight, verified = resolve_weights(record["world"], record["stage"], path)
        model = load_model(weight, device="cpu")
        if tensor_digest(model.state_dict()) != verified["model_sha256"]:
            raise ValueError("Loaded model differs from the release manifest")
        models[record["level"]] = (verified, model)
    return manifest, models


class Window:
    def __init__(self, fps):
        import os
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((1040, 824))
        pygame.display.set_caption("Mario World 1 | Locally trained policies")
        self.font = pygame.font.SysFont("Menlo", 18)
        self.small = pygame.font.SysFont("Menlo", 14)
        self.clock = pygame.time.Clock()
        self.fps, self.paused = fps, False

    def poll(self):
        restart = False
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT or (event.type == self.pg.KEYDOWN and event.key == self.pg.K_ESCAPE):
                return False, restart
            if event.type == self.pg.KEYDOWN:
                if event.key == self.pg.K_p:
                    self.paused = not self.paused
                if event.key == self.pg.K_r:
                    restart = True
        return True, restart

    def draw(self, frame, info, level, method, steps, message):
        pg = self.pg
        self.screen.fill((15, 20, 31))
        label = "PPO" if method == "ppo" else "SELF-TRAJECTORY BC"
        self.screen.blit(self.font.render(f"WORLD {level}  /  {label}  /  CPU IMAGE POLICY", True, (240, 244, 249)), (24, 24))
        surface = pg.surfarray.make_surface(np.asarray(frame).transpose(1, 0, 2))
        self.screen.blit(pg.transform.scale(surface, (768, 720)), (24, 68))
        lines = [("STATUS", "PAUSED" if self.paused else message), ("DECISIONS", str(steps)),
                 ("POSITION", str(info.get("x_pos", 0))), ("COINS", str(info.get("coins", 0))),
                 ("SCORE", str(info.get("score", 0))), ("TIME", str(info.get("time", 0)))]
        for index, (title, value) in enumerate(lines):
            y = 90 + index * 75
            self.screen.blit(self.small.render(title, True, (137, 153, 177)), (812, y))
            self.screen.blit(self.font.render(value, True, (90, 226, 173)), (812, y + 25))
        for index, text in enumerate(("P: pause", "R: restart level", "Esc: quit", "Own trained weights")):
            self.screen.blit(self.small.render(text, True, (180, 191, 208)), (812, 640 + 28 * index))
        self.screen.blit(self.small.render("Each level starts independently with its own frozen policy.", True, (137, 153, 177)), (24, 800))
        pg.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        self.pg.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "weights/manifest.json")
    parser.add_argument("--level", choices=["all", "1-1", "1-2", "1-3", "1-4"], default="all")
    parser.add_argument("--verify", action="store_true", help="Check all files and network tensors without running a game")
    parser.add_argument("--headless", action="store_true", help="Run once without opening a window")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output", type=Path, help="New directory for this separate demonstration run")
    args = parser.parse_args(argv)
    if not 1 <= args.fps <= 120:
        parser.error("fps must be between 1 and 120")
    torch.set_num_threads(1)
    manifest, models = checked_models(args.manifest)
    if args.verify:
        print(json.dumps({"status": "verified_files_and_tensors", "levels": list(models), "game_started": False}))
        return 0
    from mario_rl.env import make_env
    from mario_rl.evaluate import telemetry
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%f%z")
    output = args.output or ROOT / "outputs/world1-demo" / stamp
    output.mkdir(parents=True, exist_ok=False)
    window = None if args.headless else Window(args.fps)
    results, cancelled, issued, error = [], False, 0, None
    levels = list(models) if args.level == "all" else [args.level]
    try:
        for level in levels:
            record, model = models[level]
            retry = True
            while retry and not cancelled:
                retry = False
                env = make_env(1, record["stage"], seed=0, max_steps=5000, reward_mode="full", castle_path_rules=False)
                steps, total = 0, 0.0
                terminated = truncated = False
                try:
                    obs, info = env.reset(seed=0)
                    with (output / f"episode-{len(results)+1}-{level}.jsonl").open("x") as trace:
                        trace.write(json.dumps({"step": 0, "info": telemetry(info)}) + "\n")
                        while steps < 3000 and not (terminated or truncated):
                            if window:
                                alive, retry = window.poll()
                                if not alive or retry:
                                    cancelled = not alive
                                    break
                                if window.paused:
                                    window.draw(env.render(), info, level, record["method"], steps, "PLAYING")
                                    continue
                            with torch.inference_mode():
                                logits, _ = model(torch.from_numpy(obs).unsqueeze(0))
                                action = int(logits.argmax(1).item())
                            issued += 1
                            obs, reward, terminated, truncated, info = env.step(action)
                            steps += 1
                            total += float(info["raw_reward"])
                            trace.write(json.dumps({"step": steps, "action": action, "raw_reward": float(info["raw_reward"]),
                                "terminated": terminated, "truncated": truncated, "info": telemetry(info)}) + "\n")
                            if window:
                                window.draw(env.render(), info, level, record["method"], steps, "PLAYING")
                    clear = bool(info.get("flag_get")) and terminated and not truncated
                    result = {"level": level, "steps": steps, "raw_return": total, "clear": clear,
                              "terminated": terminated, "truncated": truncated, "restarted_by_user": retry,
                              "cancelled": cancelled, "checkpoint_sha256": record["sha256"], "method": record["method"]}
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    if window and not retry and not cancelled:
                        until = time.monotonic() + 1.5
                        while time.monotonic() < until:
                            alive, retry = window.poll()
                            cancelled = not alive
                            if cancelled or retry:
                                break
                            window.draw(env.render(), info, level, record["method"], steps, "CLEAR" if clear else "FAILED")
                finally:
                    env.close()
            if cancelled:
                break
    except BaseException as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
        raise
    finally:
        if window:
            window.close()
        receipt = {"scope": "separate_release_model_evaluation_not_historical_acceptance", "manifest": str(args.manifest),
                   "manifest_sha256": digest(args.manifest), "episodes": results,
                   "issued_environment_decisions": issued,
                   "completed_episode_decisions": sum(row["steps"] for row in results),
                   "continuous_single_life_claim": False, "cancelled": cancelled, "error": error}
        (output / "results.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return 0 if not cancelled and results and all(row["clear"] for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
