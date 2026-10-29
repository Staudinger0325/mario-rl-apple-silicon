#!/usr/bin/env python3
"""Measure process-environment sampling plus one inference pass, without training.

Run only while official training is stopped. No weights are downloaded or loaded,
and no optimizer is created. Fixed random controller actions keep CPU/MPS runs
at a given environment count on the same game trajectories.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def benchmark_worker(connection, kwargs):
    """Set worker OpenCV threads locally without editing VectorEnv's worker."""
    import cv2
    cv2.setNumThreads(1)
    from mario_rl.vector_env import _worker
    _worker(connection, kwargs)


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sysctl(name):
    result = subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def run_configuration(num_envs, device_name, steps, warmup_steps, seed):
    import gc
    import numpy as np
    import torch
    from mario_rl.model import ActorCritic
    from mario_rl.vector_env import VectorEnv

    device = torch.device(device_name)

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()

    torch.manual_seed(seed)
    startup = time.perf_counter()
    model = ActorCritic().eval().to(device)
    vector = None
    rng = np.random.default_rng(seed + 10000)
    actions = rng.integers(0, 7, size=(warmup_steps + steps, num_envs), dtype=np.int64)
    try:
        vector = VectorEnv(num_envs=num_envs, world=1, stage=1, seed=seed,
                           reward_mode="full", castle_path_rules=False,
                           backend="process", max_steps=5000, timeout=60)
        observations, _ = vector.reset()
        sync()
        startup_seconds = time.perf_counter() - startup

        def advance(controller_actions):
            nonlocal observations
            begin = time.perf_counter()
            with torch.no_grad():
                # Exactly one forward, including input transfer and small output
                # transfers analogous to recording a rollout's actions/values.
                logits, value = model(torch.from_numpy(observations).to(device))
                logits.argmax(dim=-1).cpu().numpy()
                value.cpu().numpy()
            sync()
            inference_seconds = time.perf_counter() - begin
            begin = time.perf_counter()
            observations, _, terminated, truncated, infos = vector.step(controller_actions)
            env_seconds = time.perf_counter() - begin
            finished = np.flatnonzero(terminated | truncated).tolist()
            begin = time.perf_counter()
            if finished:
                resets = vector.reset_at(finished)
                for index, (observation, _) in resets.items():
                    observations[index] = observation
            reset_seconds = time.perf_counter() - begin
            environment_steps = sum(int(info.get("frames_advanced", 0)) for info in infos)
            return inference_seconds, env_seconds, reset_seconds, len(finished), environment_steps

        warmup_start = time.perf_counter()
        for index in range(warmup_steps):
            advance(actions[index])
        sync()
        warmup_seconds = time.perf_counter() - warmup_start
        inference_total = env_total = reset_total = 0.0
        resets_total = environment_steps_total = 0
        main_cpu_start = time.process_time()
        begin = time.perf_counter()
        for index in range(warmup_steps, warmup_steps + steps):
            inference_seconds, env_seconds, reset_seconds, resets, environment_steps = advance(actions[index])
            inference_total += inference_seconds
            env_total += env_seconds
            reset_total += reset_seconds
            resets_total += resets
            environment_steps_total += environment_steps
        sync()
        elapsed = time.perf_counter() - begin
        cpu_seconds = time.process_time() - main_cpu_start
        return {
            "device": device_name,
            "num_envs": num_envs,
            "vector_steps": steps,
            "decisions": steps * num_envs,
            "elapsed_seconds": elapsed,
            "decisions_per_second": steps * num_envs / elapsed,
            "vector_steps_per_second": steps / elapsed,
            "counted_environment_steps": environment_steps_total,
            "environment_steps_per_second": environment_steps_total / elapsed,
            "inference_and_transfer_seconds": inference_total,
            "environment_step_ipc_seconds": env_total,
            "episode_reset_seconds": reset_total,
            "completed_episodes_during_measurement": resets_total,
            "main_process_cpu_seconds": cpu_seconds,
            "main_process_cpu_percent_one_core": cpu_seconds / elapsed * 100,
            "startup_seconds_excluded": startup_seconds,
            "warmup_seconds_excluded": warmup_seconds,
            "warmup_vector_steps": warmup_steps,
            "max_abs_observation": float(np.max(np.abs(observations))),
        }
    finally:
        if vector is not None:
            vector.close()
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-counts", nargs="+", type=int, default=[8, 12, 16, 24])
    parser.add_argument("--devices", nargs="+", choices=["cpu", "mps"], default=["cpu", "mps"])
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "scratch-world1" / "sampling-benchmark.json")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup_steps < 1 or any(n < 1 for n in args.env_counts):
        parser.error("environment counts, steps and warmup steps must be positive")
    sys.path.insert(0, str(ROOT))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    import cv2
    import torch
    import mario_rl.vector_env as vector_module
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if "mps" in args.devices and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is unavailable")
    # The spawned entry point is changed only in this benchmark process. The
    # library on disk and any independently running process are unaffected.
    vector_module._worker = benchmark_worker
    payload = {
        "status": "running",
        "started_at": timestamp(),
        "pid": os.getpid(),
        "hardware": {
            "machine": platform.machine(), "macos": platform.mac_ver()[0],
            "chip": sysctl("machdep.cpu.brand_string"),
            "logical_cpus": os.cpu_count(),
            "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
            "efficiency_cores": sysctl("hw.perflevel1.physicalcpu"),
            "memory_bytes": sysctl("hw.memsize"),
        },
        "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "opencv": cv2.__version__},
        "protocol": {
            "env_backend": "process", "world": 1, "stage": 1,
            "env_counts": args.env_counts, "devices": args.devices,
            "measured_vector_steps": args.steps, "warmup_vector_steps": args.warmup_steps,
            "seed": args.seed, "action_source": "fixed_seed_uniform_random",
            "initialization": "fresh_random_model_no_checkpoint",
            "forwards_per_vector_step": 1, "gradients_enabled": False,
            "optimizer_updates": 0, "torch_threads": 1, "opencv_worker_threads": 1,
            "mps_synchronize": "after each inference and at measurement boundaries",
            "includes": ["input_transfer", "one_forward", "argmax_and_value_cpu_transfer",
                         "process_environment_step_ipc", "terminal_resets"],
            "excludes": ["model_and_worker_startup", "warmup", "backward", "optimizer_updates", "evaluation"],
            "interpretation": "Sampling/inference throughput only; not end-to-end PPO throughput or learning quality.",
            "concurrency_requirement": "Run only after official training has stopped; enforced by operator.",
        },
        "results": [],
    }
    started = time.perf_counter()
    write_json(args.output, payload)
    try:
        for count in args.env_counts:
            for device in args.devices:
                print(f"Benchmark {device}, {count} process environments...", flush=True)
                result = run_configuration(count, device, args.steps, args.warmup_steps, args.seed)
                payload["results"].append(result)
                payload["updated_at"] = timestamp()
                write_json(args.output, payload)
                print(json.dumps(result, ensure_ascii=False), flush=True)
        payload["status"] = "complete"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error"] = repr(error)
        raise
    finally:
        payload["finished_at"] = timestamp()
        payload["total_wall_seconds"] = time.perf_counter() - started
        write_json(args.output, payload)


if __name__ == "__main__":
    main()
