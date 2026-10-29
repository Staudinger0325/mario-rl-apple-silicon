#!/usr/bin/env python3
"""Isolated, Go-Explore-inspired own-prefix search; no model or training.

This is not a reproduction of Go-Explore. A verified action trajectory is NOT
a validated CNN policy. Portable mode starts a new bounded experiment from an
empty archive. Legacy scheduler registrations remain a separate optional path.
Importing this module never creates NES; tests inject synthetic environments.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIONS = (("NOOP",), ("right",), ("right", "A"), ("right", "B"),
           ("right", "A", "B"), ("A",), ("left",))
PHASES = ("exploration", "prefix_replay", "trajectory_verification")


def normalized(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite evidence")
        return value
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized(item) for item in value]
    if isinstance(value, np.ndarray):
        return normalized(value.tolist())
    raise ValueError(f"Unsupported evidence type: {type(value).__name__}")


def encoded(value):
    return json.dumps(normalized(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode()


def json_hash(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Refusing a symlink evidence destination")
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(encoded(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def array_fingerprint(array):
    array = np.asarray(array)
    return {"sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "shape": list(array.shape), "dtype": array.dtype.str}


def snapshot(obs, info, wrapper=None):
    obs = np.asarray(obs)
    if (obs.shape != (4, 84, 84) or obs.dtype != np.float32
            or not np.isfinite(obs).all() or np.any(obs < 0) or np.any(obs > 1)):
        raise ValueError("Expected the original float32 (4,84,84) observation")
    info = normalized(info)
    if not isinstance(info, dict):
        raise ValueError("info must be a dictionary")
    if (info.get("world") != 1 or info.get("stage") != 3
            or info.get("castle_path_rules") is not False
            or info.get("castle_path_rule_triggered", False) is not False
            or info.get("reward_mode") != "full"):
        raise ValueError("Environment changed the registered 1-3/full/no-rules contract")
    hidden = {}
    for name in ("_steps", "_score", "_current_x", "_needs_reset", "_seed", "_closed"):
        if wrapper is not None and hasattr(wrapper, name):
            hidden[name] = normalized(getattr(wrapper, name))
    for name in ("_states", "_rgb"):
        if wrapper is not None and getattr(wrapper, name, None) is not None:
            hidden[name] = array_fingerprint(getattr(wrapper, name))
    return {"observation": array_fingerprint(obs), "info": info, "wrapper_fields": hidden}


@dataclass(frozen=True)
class SearchConfig:
    output: Path
    budget: int = 500_000
    seed: int = 42
    env_seed: int = 0
    horizon: int = 32
    max_cells: int = 256
    max_prefix: int = 3000
    max_archive_decisions: int = 65536
    curve_every: int = 256
    max_hours: float = 1.5
    deadline: str | None = None
    stop_file: Path | None = None

    def validate(self):
        for name in ("budget", "horizon", "max_cells", "max_prefix",
                     "max_archive_decisions", "curve_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_cells < 2 or self.max_archive_decisions < self.max_prefix:
            raise ValueError("Archive must hold root plus a complete permitted prefix")
        if not math.isfinite(self.max_hours) or not 0 < self.max_hours <= 8:
            raise ValueError("max_hours must be in (0,8]")
        if type(self.seed) is not int or type(self.env_seed) is not int:
            raise ValueError("Seeds must be integers")
        if self.deadline and datetime.fromisoformat(self.deadline).utcoffset() is None:
            raise ValueError("deadline requires a timezone")


class Halt(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__(status)


class EvidenceMismatch(Exception):
    pass


@dataclass
class Entry:
    prefix: tuple
    shaped_return: float
    raw_return: float
    visits: int = 0
    spatial_bin: tuple | None = None


class Archive:
    """Finite entries and a conservative bound on all retained prefix lengths."""
    def __init__(self, max_cells, max_decisions):
        self.entries = {}
        self.max_cells, self.max_decisions = max_cells, max_decisions
        self.insertions = 0

    @property
    def stored_decisions(self):
        return sum(len(entry.prefix) for entry in self.entries.values())

    def choose(self, rng):
        items = list(self.entries.items())
        if rng.random() < 0.5:
            key, entry = rng.choice(items)
        else:
            items.sort(key=lambda item: (item[1].shaped_return, item[0]))
            weights = [(rank + 1) / (entry.visits + 1)
                       for rank, (_, entry) in enumerate(items)]
            key, entry = rng.choices(items, weights=weights, k=1)[0]
        entry.visits += 1
        return key, entry

    def add(self, obs, info, prefix, shaped_return, raw_return):
        # Spatial bins organize search; no coordinates select or override actions.
        coarse = (np.asarray(obs)[-1, ::12, ::12] * 7).astype(np.uint8)
        spatial_bin = (info.get("game_area"), int(info["x_pos"]) // 32, int(info["y_pos"]) // 16)
        key = json_hash([spatial_bin, hashlib.sha256(coarse.tobytes()).hexdigest()])
        old = self.entries.get(key)
        if old and (shaped_return < old.shaped_return or
                    (shaped_return == old.shaped_return and len(prefix) >= len(old.prefix))):
            return
        self.entries[key] = Entry(tuple(prefix), shaped_return, raw_return,
                                  old.visits if old else 0, spatial_bin)
        self.insertions += int(old is None)
        while (len(self.entries) > self.max_cells
               or self.stored_decisions > self.max_decisions):
            occupancy = {}
            for item_key, entry in self.entries.items():
                if item_key != "root":
                    occupancy[entry.spatial_bin] = occupancy.get(entry.spatial_bin, 0) + 1
            crowded = max(occupancy.values())
            removable = [(entry.shaped_return, -entry.visits, -len(entry.prefix), item_key)
                         for item_key, entry in self.entries.items()
                         if item_key != "root" and occupancy[entry.spatial_bin] == crowded]
            del self.entries[min(removable)[-1]]


class Search:
    def __init__(self, config, env_factory, *, environment_kind, clock=time.time):
        config.validate()
        if environment_kind not in {"real_nes", "synthetic_fixture"}:
            raise ValueError("Explicit environment_kind is required")
        self.config, self.env_factory, self.clock = config, env_factory, clock
        self.kind = environment_kind
        self.output = Path(config.output).absolute()
        for path in (self.output, *self.output.parents):
            if path.is_symlink():
                raise ValueError("Output path may not traverse symlinks")
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = clock()
        self.deadline = self.started + config.max_hours * 3600
        if config.deadline:
            self.deadline = min(self.deadline, datetime.fromisoformat(config.deadline).timestamp())
        self.rng = random.Random(config.seed)
        self.archive = Archive(config.max_cells, config.max_archive_decisions)
        self.counts = {phase: 0 for phase in PHASES}
        self.matched = {phase: 0 for phase in PHASES if phase != "exploration"}
        self.completed = self.resets = self.attempts = self.candidates = 0
        self.search_terminal_episodes = self.search_truncated_episodes = 0
        self.max_x_by_area = {}
        self.env = None
        self.runtime_versions = {"python": sys.version.split()[0], "numpy": np.__version__}
        for package in ("gym", "nes-py", "gym-super-mario-bros"):
            try:
                self.runtime_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self.runtime_versions[package] = None
        self.current = self.root = None
        self.current_source_decision = None
        self.candidate_path = self.verification_path = None
        self.signal_requested = None
        self.status = "running"
        self.error = None
        self.events = (self.output / "events.jsonl").open("x", buffering=1)
        self.curve_file = (self.output / "search-curve.csv").open("x", newline="", buffering=1)
        self.curve = csv.DictWriter(self.curve_file, fieldnames=[
            "total_decisions", *PHASES, "completed_decisions", "elapsed_seconds",
            "archive_cells", "retained_prefix_decisions", "archive_insertions",
            "best_archived_shaped_return", "best_archived_raw_return", "max_x_by_area",
            "search_attempts", "search_terminal_episodes", "search_truncated_episodes",
            "resets", "terminal_clear_candidates", "status"])
        self.curve.writeheader()
        sources = {}
        for path in (Path(__file__), ROOT / "mario_rl/env.py"):
            data = path.read_bytes()
            sources[str(path.relative_to(ROOT))] = {
                "sha256": hashlib.sha256(data).hexdigest(), "text": data.decode()}
        write_json(self.output / "source-snapshot.json", sources)
        self.save_manifest()

    @property
    def total(self):
        return sum(self.counts.values())

    def event(self, value):
        self.events.write(encoded(value).decode() + "\n")

    def check(self, *, need_step=False):
        if self.signal_requested:
            raise Halt(self.signal_requested)
        if ((self.output / "STOP").exists()
                or (self.config.stop_file and Path(self.config.stop_file).exists())):
            raise Halt("stop_requested")
        if self.clock() >= self.deadline:
            raise Halt("deadline_reached")
        if need_step and self.total >= self.config.budget:
            raise Halt("budget_exhausted")

    def reset(self):
        self.check()
        obs, info = self.env.reset(seed=self.config.env_seed)
        self.resets += 1
        actual = snapshot(obs, info, self.env)
        self.event({"kind": "reset", "reset": self.resets, "snapshot": actual})
        if self.root is None:
            self.root = actual
            self.archive.entries["root"] = Entry((), 0.0, 0.0)
        elif encoded(actual) != encoded(self.root):
            raise EvidenceMismatch("Cold reset observation or complete info changed")
        self.current = actual
        self.current_source_decision = None
        return obs

    def step(self, action, phase, expected=None):
        self.check(need_step=True)
        if phase not in PHASES or type(action) is not int or not 0 <= action < 7:
            raise ValueError("Invalid action or accounting phase")
        # Charge issued calls, including one that raises after advancing the game.
        self.counts[phase] += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
            raise ValueError("Environment termination flags must be booleans")
        after = snapshot(obs, info, self.env)
        reward = float(reward)
        normalized(reward)
        raw = float(after["info"]["raw_reward"])
        normalized(raw)
        transition = {"action": action, "before": self.current, "after": after,
                      "reward": reward, "terminated": bool(terminated), "truncated": bool(truncated)}
        self.completed += 1
        record = {"source_decision": self.total, "transition": transition,
                  "transition_sha256": json_hash(transition)}
        evidence = {"kind": "step", "decision": self.total, "phase": phase,
                    "action": action, "transition_sha256": record["transition_sha256"],
                    "parent_source_decision": self.current_source_decision}
        if expected is None:
            evidence["transition"] = transition
        else:
            evidence.update(expected_source_decision=expected["source_decision"],
                            expected_sha256=expected["transition_sha256"],
                            matched=encoded(transition) == encoded(expected["transition"]))
            if not evidence["matched"]:
                evidence["actual_transition"] = transition
            else:
                self.matched[phase] += 1
        self.event(evidence)
        self.current = after
        self.current_source_decision = expected["source_decision"] if expected else record["source_decision"]
        area = str(after["info"].get("game_area"))
        self.max_x_by_area[area] = max(self.max_x_by_area.get(area, -math.inf), int(after["info"]["x_pos"]))
        if self.total % self.config.curve_every == 0:
            self.progress()
        if expected is not None and not evidence["matched"]:
            raise EvidenceMismatch(f"{phase} differs at source decision {expected['source_decision']}")
        return obs, record, raw

    @staticmethod
    def clear(record):
        step = record["transition"]
        flag = step["after"]["info"].get("flag_get")
        return ((flag is True or type(flag) is int and flag == 1) and step["terminated"]
                and not step["truncated"] and not step["after"]["info"].get("TimeLimit.truncated", False))

    def verify(self, prefix):
        self.candidates += 1
        self.candidate_path = self.output / "candidate-trajectory.json"
        write_json(self.candidate_path, {"scope": "self_discovered_action_trajectory_not_network",
                   "environment_kind": self.kind, "reset_seed": self.config.env_seed,
                   "reset": self.root, "steps": prefix, "terminal_clear": True})
        if self.config.budget - self.total < len(prefix):
            raise Halt("candidate_unverified_budget")
        self.verification_path = self.output / "trajectory-verification.jsonl"
        with self.verification_path.open("x", buffering=1) as stream:
            self.reset()
            stream.write(encoded({"kind": "reset", "snapshot": self.current}).decode() + "\n")
            for index, expected in enumerate(prefix):
                _, actual, _ = self.step(expected["transition"]["action"],
                                         "trajectory_verification", expected)
                stream.write(encoded(actual).decode() + "\n")
                if index < len(prefix) - 1 and (actual["transition"]["terminated"] or actual["transition"]["truncated"]):
                    raise EvidenceMismatch("Verification ended before the full trajectory")
            if not self.clear(actual):
                raise EvidenceMismatch("Full replay did not end with a real terminal clear")
        # A last step may finish after a STOP/deadline arrived. Preserve its facts,
        # but do not promote a stopped run to verified. Exact budget use is valid.
        self.check()
        self.status = "trajectory_verified"

    def progress(self):
        self.curve.writerow({"total_decisions": self.total, **self.counts,
            "completed_decisions": self.completed, "elapsed_seconds": self.clock() - self.started,
            "archive_cells": len(self.archive.entries), "retained_prefix_decisions": self.archive.stored_decisions,
            "archive_insertions": self.archive.insertions, "search_attempts": self.attempts,
            "best_archived_shaped_return": max((item.shaped_return for item in self.archive.entries.values()), default=0),
            "best_archived_raw_return": max((item.raw_return for item in self.archive.entries.values()), default=0),
            "max_x_by_area": json.dumps(self.max_x_by_area, sort_keys=True),
            "search_terminal_episodes": self.search_terminal_episodes,
            "search_truncated_episodes": self.search_truncated_episodes,
            "resets": self.resets, "terminal_clear_candidates": self.candidates, "status": self.status})
        self.save_manifest()

    def save_manifest(self, final=False):
        config = asdict(self.config)
        config["output"] = str(self.output)
        config["stop_file"] = str(self.config.stop_file) if self.config.stop_file else None
        artifacts = {}
        if final:
            for path in self.output.iterdir():
                if path.is_file() and path.name not in {"manifest.json", "STOP"}:
                    artifacts[path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        self.manifest = {"schema_version": 1, "method": "Go-Explore-inspired own-prefix random search",
            "not_a_paper_reproduction": True, "environment_kind": self.kind, "config": config,
            "environment_created": self.env is not None, "runtime_versions": self.runtime_versions,
            "started_at": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
            "recorded_at": datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(),
            "effective_deadline": datetime.fromtimestamp(self.deadline, timezone.utc).isoformat(),
            "deadline_semantics": "Cooperative checks before reset/step; a blocked call needs an external supervisor",
            "status": self.status, "error": self.error, "counts": {**self.counts,
                "total_decisions": self.total, "completed_decisions": self.completed,
                "issued_calls_without_complete_transition_record": self.total - self.completed},
            "matched_replay_transitions": self.matched,
            "counting_unit": "One issued public env.step action; includes all replay and verification calls. Reset/internal NES frames are not claimed measured.",
            "resets": self.resets, "archive_cells": len(self.archive.entries),
            "retained_prefix_decisions": self.archive.stored_decisions,
            "archive_insertions": self.archive.insertions, "archive_initial_state": "empty_before_first_actual_reset",
            "active_prefix_bound": self.config.max_prefix,
            "restoration_evidence": "Full returned float32 observation bytes, complete normalized info, reward/termination, and captured wrapper fields; not a snapshot or proof of all NES CPU/PPU/RNG internals",
            "search_attempts": self.attempts, "max_x_by_area": self.max_x_by_area,
            "search_terminal_episodes": self.search_terminal_episodes,
            "search_truncated_episodes": self.search_truncated_episodes,
            "terminal_clear_candidates": self.candidates,
            "unverified_candidate": self.candidates > 0 and self.status != "trajectory_verified",
            "trajectory_verified": self.status == "trajectory_verified",
            "candidate_trajectory": self.candidate_path.name if self.candidate_path else None,
            "verification_trace": self.verification_path.name if self.verification_path else None,
            "actions": ACTIONS, "action_selection": "uniform seven actions; uniform holds from [1,2,4,8] decisions",
            "archive_selection": "half uniform; half cumulative-shaped-return rank divided by visits+1",
            "archive_retention": "Evict from the most populated (area,x//32,y//16) bin first; lower return/visited/longer prefixes break ties. One representative per bin is retained while hard cell/prefix limits permit.",
            "model_created": False, "gradient_updates": 0, "network_policy_validated": False,
            "project_complete": False, "artifacts": artifacts}
        write_json(self.output / "manifest.json", self.manifest)

    def run(self):
        try:
            self.check()
            self.env = self.env_factory()
            if self.env.action_space.n != 7 or tuple(map(tuple, self.env.actions)) != ACTIONS:
                raise ValueError("Expected the unchanged seven SIMPLE_MOVEMENT actions")
            self.reset()
            while self.status == "running":
                self.check(need_step=True)
                self.attempts += 1
                cell, entry = self.archive.choose(self.rng)
                self.event({"kind": "attempt", "attempt": self.attempts, "cell": cell,
                            "prefix_source_decisions": [item["source_decision"] for item in entry.prefix]})
                obs = self.reset()
                prefix = list(entry.prefix)
                shaped, raw_return = entry.shaped_return, entry.raw_return
                for expected in prefix:
                    obs, _, _ = self.step(expected["transition"]["action"], "prefix_replay", expected)
                hold = 0
                for _ in range(min(self.config.horizon, self.config.max_prefix - len(prefix))):
                    if hold == 0:
                        action = self.rng.randrange(7)
                        hold = self.rng.choice((1, 2, 4, 8))
                    hold -= 1
                    obs, record, raw = self.step(action, "exploration")
                    prefix.append(record)
                    shaped += record["transition"]["reward"]
                    raw_return += raw
                    step = record["transition"]
                    self.search_terminal_episodes += int(step["terminated"])
                    self.search_truncated_episodes += int(step["truncated"])
                    if self.clear(record):
                        self.verify(prefix)
                        break
                    if step["terminated"] or step["truncated"]:
                        break
                    if len(prefix) < self.config.max_prefix:
                        self.archive.add(obs, self.current["info"], prefix, shaped, raw_return)
                self.progress()
        except Halt as error:
            self.status = error.status
        except KeyboardInterrupt:
            self.status = "interrupted"
        except Exception as error:
            self.status = "evidence_mismatch" if isinstance(error, EvidenceMismatch) else "error"
            self.error = f"{type(error).__name__}: {error}"
            self.event({"kind": "error", "total_decisions": self.total, "error": self.error})
        finally:
            if self.env is not None:
                try:
                    self.env.close()
                except Exception as error:
                    self.status, self.error = "error", f"Environment close failed: {error}"
            self.progress()
            self.events.close()
            self.curve_file.close()
            self.save_manifest(final=True)
        return self.manifest


def checked_reference(reference, label):
    if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str)
            or not isinstance(reference.get("sha256"), str)):
        raise ValueError(f"{label} requires an actual path and SHA256")
    path = Path(reference["path"]).resolve(strict=True)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError(f"{label} bytes do not match the registered SHA256")
    return path, data


def parallel_launch_evidence(review, config):
    """Validate the opt-in amended gate; claims of live state remain the scheduler's review."""
    config.validate()
    if not isinstance(config.deadline, str) or not config.deadline.strip():
        raise ValueError("Parallel search requires a nonempty timezone-aware absolute round deadline")
    required = ("original_seed42_and_seed43_complete_failures_reviewed", "entropy01_still_running",
                "all_candidate_histories_reviewed", "free_global_job_slot_verified")
    if any(review.get(key) is not True for key in required):
        raise ValueError("Parallel search requires both original failures, running 0.1, all histories and a free slot review")
    if review.get("entropy01_complete_failure_reviewed") not in (None, False):
        raise ValueError("Parallel launch may not claim the still-running entropy0.1 experiment failed")
    for name in ("deadline", "absolute_deadline"):
        if name in review and review[name] != config.deadline:
            raise ValueError("Launch review deadline differs from actual CLI deadline")
    path, raw = checked_reference(review.get("amendment_registration"), "amendment_registration")
    amendment = json.loads(raw)
    if (not isinstance(amendment, dict) or amendment.get("format_version") != 2
            or amendment.get("decision_mode") != "parallel_after_two_original_recipe_failures"
            or amendment.get("comparison_status", {}).get("entropy01") != "still_running_at_launch_review_not_judged_failed"):
        raise ValueError("Amendment must explicitly register the V2 parallel mode without judging entropy0.1 failed")
    script_path, _ = checked_reference(amendment.get("script"), "amended script")
    if script_path != Path(__file__).resolve():
        raise ValueError("Amendment must bind the executing search script")
    old_path, old_bytes = checked_reference(amendment.get("supersedes"), "superseded registration")
    archives = {}
    for name, old_key, filename in (
            ("superseded_script_archive", "script", "superseded-search-script.py"),
            ("superseded_tests_archive", "synthetic_tests", "superseded-search-tests.py")):
        if name in amendment:
            _, archive_bytes = checked_reference(amendment[name], name)
            old = json.loads(old_bytes)
            if amendment[name]["sha256"] != old.get(old_key, {}).get("sha256"):
                raise ValueError(f"{name} does not match the original registration")
            archives[filename] = archive_bytes
    expected = asdict(config)
    expected["output"] = str(Path(config.output).absolute())
    expected["stop_file"] = str(Path(config.stop_file).absolute()) if config.stop_file else None
    actual = amendment.get("config")
    if not isinstance(actual, dict):
        raise ValueError("Amendment requires the complete effective SearchConfig")
    actual = {"stop_file": None, **actual}
    if encoded(actual) != encoded(expected):
        raise ValueError("Amended configuration differs from actual CLI arguments")
    return {"path": path, "bytes": raw, "superseded_path": old_path,
            "superseded_bytes": old_bytes, "archives": archives}


def portable_deadline(max_hours, deadline=None):
    """Give a new foreground experiment a bounded, non-expired wall budget."""
    if isinstance(max_hours, bool) or not math.isfinite(max_hours) or not 0 < max_hours <= 8:
        raise ValueError("max_hours must be in (0,8]")
    current = datetime.now(timezone.utc)
    requested = datetime.fromisoformat(deadline) if deadline else current + timedelta(hours=max_hours)
    if requested.utcoffset() is None or not math.isfinite(requested.timestamp()) or requested <= current:
        raise ValueError("deadline must be a future timezone-aware timestamp")
    return min(requested, current + timedelta(hours=max_hours)).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New isolated directory; never reuse a run")
    parser.add_argument("--budget", type=int, default=500_000, help="Total issued decisions INCLUDING replay and verification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--max-cells", type=int, default=256)
    parser.add_argument("--max-prefix", type=int, default=3000)
    parser.add_argument("--max-archive-decisions", type=int, default=65536)
    parser.add_argument("--curve-every", type=int, default=256)
    parser.add_argument("--max-hours", type=float, default=1.5)
    parser.add_argument("--deadline", help="Timezone-aware deadline; portable mode defaults to now plus max-hours")
    parser.add_argument("--portable", action="store_true", help="Start an independent new experiment without historical scheduler prerequisites")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--launch-review", type=Path,
                        help="Root-written JSON approving this conditional search after candidate/history review")
    args = parser.parse_args(argv)
    if args.portable:
        if args.launch_review is not None:
            parser.error("--portable and --launch-review are mutually exclusive")
        args.deadline = portable_deadline(args.max_hours, args.deadline)
        review = {"decision": "portable_new_experiment", "created_at": datetime.now(timezone.utc).isoformat(),
                  "scope": "fresh_empty_archive_current_source", "historical_reproduction_claim": False}
        review_bytes = encoded(review) + b"\n"
    else:
        if args.launch_review is None:
            parser.error("Use --portable for a new experiment, or supply a historical --launch-review")
        review_bytes = args.launch_review.read_bytes()
        review = json.loads(review_bytes)
    config = SearchConfig(**{key: value for key, value in vars(args).items()
                             if key not in {"launch_review", "portable"}})
    config.validate()
    amendment = None
    if not args.portable and review.get("decision") == "approved_parallel_own_prefix_search":
        try:
            amendment = parallel_launch_evidence(review, config)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    elif not args.portable and (review.get("decision") != "approved_own_prefix_search"
          or review.get("entropy01_complete_failure_reviewed") is not True
          or review.get("all_candidate_histories_reviewed") is not True):
        parser.error("An explicit scheduler launch review is required; preparation is not approval")

    def env_factory():
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from mario_rl.env import make_env
        return make_env(1, 3, seed=config.env_seed, max_steps=config.max_prefix,
                        reward_mode="full", castle_path_rules=False)

    search = Search(config, env_factory, environment_kind="real_nes")
    (search.output / "launch-review.json").write_bytes(review_bytes)
    if amendment is not None:
        (search.output / "launch-amendment-registration.json").write_bytes(amendment["bytes"])
        (search.output / "superseded-registration.json").write_bytes(amendment["superseded_bytes"])
        for name, data in amendment["archives"].items():
            (search.output / name).write_bytes(data)
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda number, frame: setattr(search, "signal_requested", f"signal_{number}"))
    try:
        result = search.run()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(json.dumps({key: result[key] for key in ("status", "counts", "trajectory_verified")}, ensure_ascii=False))
    return 0 if result["trajectory_verified"] else 1 if result["status"] in {"error", "evidence_mismatch"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
