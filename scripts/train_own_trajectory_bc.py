#!/usr/bin/env python3
"""Fit a fresh actor to one registered, locally discovered trajectory.

Dataset agreement is not closed-loop policy validation. This command never runs
a policy in NES: its only environment calls strictly replay the registered data.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
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
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import explore_own_prefixes as search
from mario_rl.model import ActorCritic, select_device

STAGE = "own_trajectory_bc"
RECIPE = {"seed": 42, "env_seed": 0, "device": "mps", "batch_size": 64,
          "lr": 0.001, "grad_clip": 0.5, "max_epochs": 500,
          "perfect_epochs": 5, "max_hours": 0.75}
METRICS = ["epoch", "completed_epoch", "train_mean_ce", "cpu_mean_ce", "cpu_correct", "dataset_size",
           "cpu_accuracy", "consecutive_perfect_epochs", "epoch_optimizer_steps", "epoch_gradient_samples",
           "bc_optimizer_steps", "bc_minibatch_samples", "gradient_norm_mean", "gradient_norm_max",
           "epoch_seconds", "elapsed_seconds", "replay_env_step_calls", "ppo_decisions"]


class Halt(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def plain_file(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"Expected a regular file without symlink traversal: {path}")
    return path


def reference(path):
    path = plain_file(path)
    return {"path": str(path), "sha256": search.file_hash(path), "bytes": path.stat().st_size}


def checked(ref):
    if not isinstance(ref, dict) or not {"path", "sha256"} <= set(ref):
        raise ValueError("Evidence requires path and sha256")
    path = plain_file(ref["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError(f"Registered bytes changed: {path}")
    return path, raw


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def save_checkpoint(path, payload):
    # Only our last.pt is replaced; initial.pt and final.pt have exclusive names.
    path = Path(path)
    if path.is_symlink() or (path.name != "last.pt" and path.exists()):
        raise ValueError(f"Refusing to replace checkpoint: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_names():
    return {"scripts/train_own_trajectory_bc.py", "scripts/explore_own_prefixes.py"} | {
        str(path.relative_to(ROOT)) for path in (ROOT / "mario_rl").rglob("*.py")}


def registration_recipe(reg, environment_kind):
    """Keep learning hyperparameters fixed; bind portable seed/device/budget explicitly."""
    recipe = reg.get("recipe", {})
    if set(recipe) != set(RECIPE) | {"deadline"}:
        raise ValueError("Recipe must explicitly specify all BC settings and deadline")
    portable = reg.get("execution_mode") == "portable"
    if reg.get("execution_mode") not in (None, "portable"):
        raise ValueError("Unknown execution mode")
    expected = dict(RECIPE)
    if portable:
        for key in ("seed", "env_seed"):
            if type(recipe[key]) is not int or not 0 <= recipe[key] < 2**32:
                raise ValueError("Portable seeds must be integers in [0,2**32)")
        if recipe["device"] not in {"cpu", "mps", "cuda", "auto"}:
            raise ValueError("Unsupported training device")
        hours = recipe["max_hours"]
        if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not math.isfinite(hours) or not 0 < hours <= 8:
            raise ValueError("max_hours must be in (0,8]")
        for key in ("seed", "env_seed", "device", "max_hours"):
            expected[key] = recipe[key]
    if environment_kind == "synthetic_fixture":
        expected["device"] = "cpu"
    for key, value in expected.items():
        if type(recipe[key]) is not type(value) or recipe[key] != value:
            raise ValueError(f"Unregistered BC recipe change: {key}")
    deadline = datetime.fromisoformat(recipe["deadline"])
    if deadline.utcoffset() is None or not math.isfinite(deadline.timestamp()):
        raise ValueError("Deadline must have a timezone and finite timestamp")
    return recipe, portable


def write_new_registration(path, value, validate):
    """Publish a new checked registration; never replace an existing request."""
    path = Path(path).absolute()
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Registration must be a new path without symlink traversal")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    try:
        validate(path)
    except BaseException:
        path.unlink()
        raise
    return reference(path)


def prepare_from_search(search_directory, registration, output, *, device="mps", max_hours=0.75,
                        deadline=None, environment_kind="real_nes"):
    """Register actual current-source search evidence, without a game or training call."""
    folder = plain_file(Path(search_directory) / "manifest.json").parent
    manifest = json.loads((folder / "manifest.json").read_text())
    names = {"manifest": "manifest.json", "candidate": manifest.get("candidate_trajectory"),
             "verification": manifest.get("verification_trace"), "source_snapshot": "source-snapshot.json"}
    if any(not isinstance(name, str) or Path(name).name != name for name in names.values()):
        raise ValueError("Search must provide its own verified artifact filenames")
    refs = {key: reference(folder / name) for key, name in names.items()}
    candidate = json.loads((folder / names["candidate"]).read_text())
    config = manifest.get("config", {})
    reg = {"schema_version": 1, "scope": STAGE, "execution_mode": "portable",
           "created_at": now(), "historical_reproduction_claim": False,
           "output": str(Path(output).absolute()), "trajectory_steps": len(candidate.get("steps", [])),
           "recipe": {**RECIPE, "seed": config.get("seed"), "env_seed": config.get("env_seed"),
                      "device": device, "max_hours": max_hours,
                      "deadline": search.portable_deadline(max_hours, deadline)},
           "sources": {name: search.file_hash(plain_file(ROOT / name)) for name in source_names()},
           "search": refs}
    return write_new_registration(registration, reg,
        lambda path: validate_registration(path, environment_kind=environment_kind))


def validate_registration(path, *, environment_kind="real_nes"):
    if environment_kind not in {"real_nes", "synthetic_fixture"}:
        raise ValueError("Explicit supported environment kind required")
    path = plain_file(path)
    raw = path.read_bytes()
    reg = json.loads(raw)
    if reg.get("schema_version") != 1 or reg.get("scope") != STAGE:
        raise ValueError("Expected schema_version=1 and own_trajectory_bc scope")
    recipe, portable = registration_recipe(reg, environment_kind)
    n = reg.get("trajectory_steps")
    if type(n) is not int or n < 1 or (environment_kind == "real_nes" and not portable and n != 439):
        raise ValueError("This registered real pilot uses exactly the seed42 439-step trajectory")
    sources = reg.get("sources")
    if not isinstance(sources, dict) or set(sources) != source_names():
        raise ValueError("Source registration must cover both scripts and all Mario Python files")
    source_snapshot = {}
    for name, expected_hash in sources.items():
        data = plain_file(ROOT / name).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_hash:
            raise ValueError(f"Current source differs from registration: {name}")
        source_snapshot[name] = {"sha256": actual, "text": data.decode("utf-8")}
    search_refs = reg.get("search", {})
    if set(search_refs) != {"manifest", "candidate", "verification", "source_snapshot"}:
        raise ValueError("Search evidence must bind manifest, candidate, verification, and source snapshot")
    captured = {key: checked(ref) for key, ref in search_refs.items()}
    manifest = json.loads(captured["manifest"][1])
    candidate = json.loads(captured["candidate"][1])
    verified = [json.loads(line) for line in captured["verification"][1].splitlines()]
    old_sources = json.loads(captured["source_snapshot"][1])
    if (manifest.get("status") != "trajectory_verified" or manifest.get("trajectory_verified") is not True
            or manifest.get("environment_kind") != environment_kind
            or manifest.get("model_created") is not False or manifest.get("gradient_updates") != 0
            or manifest.get("config", {}).get("seed") != recipe["seed"]
            or manifest["config"].get("env_seed") != recipe["env_seed"]
            or type(manifest["config"].get("max_prefix")) is not int or n > manifest["config"]["max_prefix"]
            or manifest.get("matched_replay_transitions", {}).get("trajectory_verification") != n):
        raise ValueError("Search is not the registered verified, model-free seed42 discovery")
    folder = captured["manifest"][0].parent
    for key in ("candidate", "verification", "source_snapshot"):
        source_path, data = captured[key]
        artifact = manifest.get("artifacts", {}).get(source_path.name, {})
        if source_path.parent != folder or artifact.get("sha256") != hashlib.sha256(data).hexdigest() or artifact.get("bytes") != len(data):
            raise ValueError("Search manifest does not bind the supplied artifact")
    if (manifest.get("candidate_trajectory") != captured["candidate"][0].name
            or manifest.get("verification_trace") != captured["verification"][0].name):
        raise ValueError("Search artifact identities differ")
    if set(old_sources) != {"scripts/explore_own_prefixes.py", "mario_rl/env.py"}:
        raise ValueError("Unexpected search source snapshot")
    for name, record in old_sources.items():
        if (hashlib.sha256(record["text"].encode()).hexdigest() != record["sha256"]
                or record["sha256"] != sources[name]):
            raise ValueError("Search and BC environment/source behavior differ")
    if (candidate.get("scope") != "self_discovered_action_trajectory_not_network"
            or candidate.get("environment_kind") != environment_kind or candidate.get("reset_seed") != recipe["env_seed"]
            or candidate.get("terminal_clear") is not True or len(candidate.get("steps", [])) != n
            or len(verified) != n + 1
            or verified[0] != {"kind": "reset", "snapshot": candidate.get("reset")}):
        raise ValueError("Candidate and cold verification are incomplete or differ")
    before = candidate["reset"]
    labels = {}
    for index, (record, replay) in enumerate(zip(candidate["steps"], verified[1:])):
        transition = record.get("transition", {})
        action = transition.get("action")
        if (type(action) is not int or not 0 <= action < 7
                or transition.get("before") != before
                or record.get("transition_sha256") != search.json_hash(transition)
                or replay.get("transition_sha256") != record["transition_sha256"]
                or replay.get("transition") != transition):
            raise ValueError("Broken or altered trajectory transition")
        fp = transition["before"]["observation"]
        key = search.json_hash(fp)
        if key in labels and labels[key] != action:
            raise ValueError("Identical registered observations have conflicting action labels")
        labels[key] = action
        if index < n - 1 and (transition.get("terminated") or transition.get("truncated")):
            raise ValueError("Trajectory terminates before its final step")
        before = transition["after"]
    if not search.Search.clear(candidate["steps"][-1]):
        raise ValueError("Trajectory does not finish with an actual non-truncated terminal clear")
    output = Path(reg["output"]).absolute()
    if output.exists() or any(item.is_symlink() for item in (output, *output.parents)):
        raise ValueError("BC output must be a new directory without symlink traversal")
    if output == folder or output in folder.parents or folder in output.parents:
        raise ValueError("BC output must be isolated from search evidence")
    return {"registration": reg, "registration_raw": raw,
            "registration_ref": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()},
            "captured": captured, "manifest": manifest, "candidate": candidate,
            "source_snapshot": source_snapshot, "output": output, "environment_kind": environment_kind}


class BC:
    def __init__(self, context, *, env_factory=None, model_factory=None, clock=time.time):
        if context["environment_kind"] == "real_nes" and (env_factory is not None or model_factory is not None):
            raise ValueError("Injected environments/models require explicit synthetic fixture mode")
        self.context, self.clock = context, clock
        self.reg, self.output = context["registration"], context["output"]
        self.recipe = self.reg["recipe"]
        self.started = clock()
        self.deadline = min(self.started + self.recipe["max_hours"] * 3600,
                            datetime.fromisoformat(self.recipe["deadline"]).timestamp())
        self.env_factory, self.model_factory = env_factory, model_factory or ActorCritic
        self.env = self.model = self.optimizer = None
        self.signal_requested = None
        self.status, self.error = "initializing", None
        self.issued = self.returned = self.matched = self.resets = 0
        self.epoch = self.optimizer_steps = self.optimizer_calls = self.minibatch_samples = self.perfect = 0
        self.dataset_ref = self.initial_ref = self.initial_hash = None
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "registration.json").write_bytes(context["registration_raw"])
        inputs = self.output / "search-evidence"
        inputs.mkdir()
        self.input_refs = {}
        for key, (path, raw) in context["captured"].items():
            destination = inputs / path.name
            destination.write_bytes(raw)
            self.input_refs[key] = reference(destination)
        search.write_json(self.output / "source-snapshot.json", context["source_snapshot"])
        self.source_ref = reference(self.output / "source-snapshot.json")
        self.persist()

    def check(self):
        if self.signal_requested:
            raise Halt(self.signal_requested)
        if (self.output / "STOP").exists() or (self.reg.get("stop_file") and Path(self.reg["stop_file"]).exists()):
            raise Halt("stop_requested")
        if self.clock() >= self.deadline:
            raise Halt("deadline_reached")

    def state(self):
        return {"schema_version": 1, "stage": STAGE, "status": self.status, "recorded_at": now(),
                "model_config": {"world": 1, "stage": 3, "seed": self.recipe["seed"], "env_seed": self.recipe["env_seed"],
                    "num_inputs": 4, "num_actions": 7, "observation_shape": [4, 84, 84],
                    "reward_mode": "full", "castle_path_rules": False, "action_repeat": 4,
                    "max_environment_steps": self.context["manifest"]["config"]["max_prefix"]},
                "environment_kind": self.context["environment_kind"], "recipe": self.recipe,
                "registration_ref": self.context["registration_ref"], "search_refs": self.input_refs,
                "source_snapshot_ref": self.source_ref, "initial_model_sha256": self.initial_hash,
                "initial_checkpoint_ref": self.initial_ref, "dataset_ref": self.dataset_ref,
                "epoch": self.epoch, "bc_optimizer_steps": self.optimizer_steps,
                "bc_optimizer_step_calls": self.optimizer_calls,
                "bc_minibatch_samples": self.minibatch_samples, "consecutive_perfect_epochs": self.perfect,
                "ppo_decisions": 0, "ppo_updates": 0,
                "replay_env_step_calls": self.issued, "returned_env_steps": self.returned,
                "matched_transitions": self.matched, "reset_calls": self.resets,
                "unreturned_env_step_calls": self.issued - self.returned,
                "counting_unit": "One issued public env.step; reset/internal NES frames are not claimed measured.",
                "search_environment_decisions": self.context["manifest"]["counts"]["total_decisions"],
                "elapsed_seconds": self.clock() - self.started,
                "effective_deadline": datetime.fromtimestamp(self.deadline, timezone.utc).isoformat(),
                "network_policy_validated": False, "project_complete": False,
                "dataset_fit_criterion_met": self.status == "dataset_fit_criterion_met", "error": self.error}

    def persist(self):
        search.write_json(self.output / "manifest.json", self.state())

    def collect(self):
        self.status = "collecting_registered_trajectory"
        self.check()
        if self.env_factory is None:
            from mario_rl.env import make_env
            self.env = make_env(world=1, stage=3, seed=self.recipe["env_seed"], reward_mode="full", castle_path_rules=False,
                                max_steps=self.context["manifest"]["config"]["max_prefix"])
        else:
            self.env = self.env_factory()
        if self.env.action_space.n != 7 or tuple(map(tuple, self.env.actions)) != search.ACTIONS:
            raise ValueError("Environment action mapping differs")
        self.check()
        self.resets += 1
        obs, info = self.env.reset(seed=self.recipe["env_seed"])
        current = search.snapshot(obs, info, self.env)
        candidate = self.context["candidate"]
        if current != candidate["reset"]:
            raise search.EvidenceMismatch("Cold-reset snapshot differs")
        observations, actions, labels = [], [], {}
        with (self.output / "dataset-replay.jsonl").open("x", buffering=1) as stream:
            stream.write(search.encoded({"kind": "reset", "snapshot": current}).decode() + "\n")
            for index, expected in enumerate(candidate["steps"]):
                self.check()
                action = expected["transition"]["action"]
                data = np.asarray(obs).copy(order="C")
                key = data.tobytes()
                if key in labels and labels[key] != action:
                    raise ValueError("Identical actual inputs have conflicting action labels")
                labels[key] = action
                observations.append(data)
                actions.append(action)
                self.issued += 1
                self.persist()  # Charge even an env.step that raises after advancing.
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.returned += 1
                if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
                    raise ValueError("Termination flags must be booleans")
                after = search.snapshot(obs, info, self.env)
                transition = {"action": action, "before": current, "after": after,
                              "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated)}
                matched = search.encoded(transition) == search.encoded(expected["transition"])
                stream.write(search.encoded({"step": index + 1, "transition": transition,
                    "transition_sha256": search.json_hash(transition),
                    "expected_source_decision": expected["source_decision"], "matched": matched}).decode() + "\n")
                if not matched:
                    raise search.EvidenceMismatch(f"Cold replay differs at decision {index + 1}")
                self.matched += 1
                current = after
            self.check()  # A final call crossing STOP/deadline is not promoted.
        self.env.close()
        self.env = None
        x, y = np.stack(observations), np.asarray(actions, dtype=np.int64)
        if x.dtype != np.float32 or x.shape != (len(actions), 4, 84, 84):
            raise ValueError("Dataset lost original float32 observation representation")
        np.save(self.output / "observations.npy", x, allow_pickle=False)
        np.save(self.output / "actions.npy", y, allow_pickle=False)
        dataset = {"schema_version": 1, "rows": len(y), "observations": reference(self.output / "observations.npy"),
                   "actions": reference(self.output / "actions.npy"), "replay_trace": reference(self.output / "dataset-replay.jsonl"),
                   "shape": list(x.shape), "dtype": str(x.dtype), "unique_inputs": len(labels),
                   "duplicate_same_action_inputs": len(y) - len(labels), "conflicting_inputs": 0,
                   "input_semantics": "Actual observation BEFORE each registered action; final post-action observation is not a training input.",
                   "registration_ref": self.context["registration_ref"], "search_refs": self.input_refs,
                   "environment_kind": self.context["environment_kind"], "replay_env_step_calls": self.issued,
                   "matched_transitions": self.matched, "actual_terminal_clear": True,
                   "network_policy_validated": False, "ppo_decisions": 0}
        search.write_json(self.output / "dataset.json", dataset)
        self.dataset_ref = reference(self.output / "dataset.json")
        return torch.from_numpy(x), torch.from_numpy(y)

    def initialize(self):
        self.check()
        torch.set_num_threads(1)
        random.seed(self.recipe["seed"])
        np.random.seed(self.recipe["seed"])
        torch.manual_seed(self.recipe["seed"])
        self.model = self.model_factory().cpu()
        self.initial_hash = model_hash(self.model)
        after_first = torch.get_rng_state().clone()
        torch.manual_seed(self.recipe["seed"])
        rebuilt = self.model_factory().cpu()
        independent_hash = model_hash(rebuilt)
        torch.set_rng_state(after_first)
        if independent_hash != self.initial_hash:
            raise ValueError("Independent seeded initialization differs")
        if not hasattr(self.model, "critic_linear"):
            raise ValueError("Expected the registered actor-critic architecture")
        self.critic_initial = cpu_tree(self.model.critic_linear.state_dict())
        for parameter in self.model.critic_linear.parameters():
            parameter.requires_grad_(False)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.recipe["lr"])
        assert not self.optimizer.state
        self.status = "initialized_fresh_scratch"
        self.checkpoint("initial.pt")
        self.initial_ref = reference(self.output / "initial.pt")
        search.write_json(self.output / "initialization-verification.json", {
            "initial_model_sha256": self.initial_hash, "independent_cpu_model_sha256": independent_hash,
            "initial_checkpoint_ref": self.initial_ref, "optimizer_state_empty": True, "seed": self.recipe["seed"],
            "source_checkpoint": None, "pretrained_used": False, "dataset_ref": self.dataset_ref,
            "environment_kind": self.context["environment_kind"], "ppo_decisions": 0})
        del rebuilt
        self.device = select_device(self.recipe["device"])
        self.model.to(self.device)
        # Device conversion may replace Parameter objects; never retain CPU Adam bindings.
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.recipe["lr"])
        actual_parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        if len(actual_parameters) != len(self.parameters) or any(a is not b for a, b in zip(actual_parameters, self.parameters)):
            raise ValueError("Adam is not bound to actual trainable model parameters")
        if self.optimizer.state:
            raise ValueError("Training-device Adam must start empty")
        if self.device.type == "mps":
            torch.mps.manual_seed(self.recipe["seed"])
        self.cpu_model = copy.deepcopy(self.model).cpu().eval()
        for parameter in self.cpu_model.parameters():
            parameter.requires_grad_(False)

    def checkpoint(self, name):
        if self.model is None:
            return
        state = self.state()
        state.update({"model": cpu_tree(self.model.state_dict()), "optimizer": cpu_tree(self.optimizer.state_dict()),
                      "model_sha256": model_hash(self.model), "rng": {"torch": torch.get_rng_state(),
                          "mps": torch.mps.get_rng_state().cpu() if getattr(self, "device", torch.device("cpu")).type == "mps" else None},
                      "lineage": {"initialization": {"kind": "scratch", "seed": self.recipe["seed"], "source_path": None, "source_sha256": None},
                                  "root_initial_model_sha256": self.initial_hash,
                                  "stages": ["fresh_scratch", "own_trajectory_bc"],
                                  "registration_ref": self.context["registration_ref"], "dataset_ref": self.dataset_ref,
                                  "initial_checkpoint_ref": self.initial_ref, "source_snapshot_ref": self.source_ref,
                                  "search_refs": self.input_refs, "ppo_decisions": 0,
                                  "critic_training": "Critic output head frozen/excluded from Adam; shared trunk learns only actor CE. No value objective."}})
        save_checkpoint(self.output / name, state)

    def accuracy(self, x, y):
        self.cpu_model.load_state_dict(cpu_tree(self.model.state_dict()), strict=True)
        correct, total_loss = 0, 0.0
        with torch.no_grad():
            for start in range(0, len(y), self.recipe["batch_size"]):
                self.check()
                logits, _ = self.cpu_model(x[start:start + self.recipe["batch_size"]])
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite CPU evaluation logits")
                target = y[start:start + len(logits)]
                correct += int((logits.argmax(1) == target).sum())
                total_loss += float(F.cross_entropy(logits, target, reduction="sum"))
        return correct, total_loss / len(y)

    def fit(self, x, y):
        self.status = "training_bc"
        with (self.output / "metrics.csv").open("x", newline="", buffering=1) as stream:
            writer = csv.DictWriter(stream, fieldnames=METRICS)
            writer.writeheader()
            for epoch in range(1, self.recipe["max_epochs"] + 1):
                self.check()
                self.epoch = epoch
                began = self.clock()
                norms, samples, updates, loss_sum = [], 0, 0, 0.0
                correct = eval_loss = None
                complete = False
                try:
                    order = torch.randperm(len(y))
                    self.model.train()
                    for start in range(0, len(y), self.recipe["batch_size"]):
                        self.check()
                        indices = order[start:start + self.recipe["batch_size"]]
                        self.optimizer.zero_grad(set_to_none=True)
                        logits, _ = self.model(x[indices].to(self.device))
                        loss = F.cross_entropy(logits, y[indices].to(self.device))
                        if not bool(torch.isfinite(loss).cpu()):
                            raise FloatingPointError("Non-finite actor CE")
                        loss.backward()
                        samples += len(indices)
                        self.minibatch_samples += len(indices)
                        loss_sum += float(loss.detach().cpu()) * len(indices)
                        norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, self.recipe["grad_clip"], error_if_nonfinite=True).cpu())
                        self.optimizer_calls += 1
                        self.optimizer.step()
                        updates += 1
                        self.optimizer_steps += 1
                        norms.append(norm)
                    correct, eval_loss = self.accuracy(x, y)
                    self.check()
                    self.perfect = self.perfect + 1 if correct == len(y) else 0
                    complete = True
                finally:
                    writer.writerow(dict(epoch=epoch, completed_epoch=complete,
                        train_mean_ce=loss_sum / samples if samples else "", cpu_mean_ce=eval_loss if eval_loss is not None else "",
                        cpu_correct=correct if correct is not None else "", dataset_size=len(y),
                        cpu_accuracy=correct / len(y) if correct is not None else "", consecutive_perfect_epochs=self.perfect,
                        epoch_optimizer_steps=updates, epoch_gradient_samples=samples,
                        bc_optimizer_steps=self.optimizer_steps, bc_minibatch_samples=self.minibatch_samples,
                        gradient_norm_mean=sum(norms) / len(norms) if norms else "", gradient_norm_max=max(norms) if norms else "",
                        epoch_seconds=self.clock() - began, elapsed_seconds=self.clock() - self.started,
                        replay_env_step_calls=self.issued, ppo_decisions=0))
                    stream.flush()
                self.checkpoint("last.pt")
                self.persist()
                print(f"epoch={epoch} CE={loss_sum/samples:.6f} CPU_accuracy={correct}/{len(y)} perfect={self.perfect}/5 updates={self.optimizer_steps}", flush=True)
                self.check()
                if self.perfect >= self.recipe["perfect_epochs"]:
                    self.status = "dataset_fit_criterion_met"
                    return
            self.check()
            self.status = "epoch_budget_exhausted"

    def run(self):
        try:
            x, y = self.collect()
            self.initialize()
            self.fit(x, y)
            self.check()  # Do not label a late STOP/deadline as fitted success.
        except Halt as error:
            self.status = str(error)
        except KeyboardInterrupt:
            self.status = "keyboard_interrupt"
        except Exception as error:
            self.status = "evidence_mismatch" if isinstance(error, search.EvidenceMismatch) else "error"
            self.error = {"type": type(error).__name__, "message": str(error)}
        finally:
            if self.env is not None:
                try:
                    self.env.close()
                except Exception as error:
                    if self.error is None:
                        self.status, self.error = "error", {"type": type(error).__name__, "message": str(error), "phase": "env.close"}
                self.env = None
            if self.status == "dataset_fit_criterion_met":
                try:
                    self.check()
                except Halt as error:
                    self.status = str(error)
            if self.model is not None and self.optimizer is not None and hasattr(self, "critic_initial"):
                if any(not torch.equal(value.cpu(), self.critic_initial[key]) for key, value in self.model.critic_linear.state_dict().items()):
                    self.status, self.error = "error", {"type": "CriticChanged", "message": "Frozen critic head changed"}
                self.checkpoint("last.pt")
                self.checkpoint("final.pt")
            self.persist()
        return self.state()


def run_bc(registration, *, environment_kind="real_nes", env_factory=None, model_factory=None, clock=time.time):
    context = validate_registration(registration, environment_kind=environment_kind)
    engine = BC(context, env_factory=env_factory, model_factory=model_factory, clock=clock)
    return engine.run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--prepare-from-search", type=Path, help="Create a new registration from an actually verified search; do not train")
    parser.add_argument("--output", type=Path, help="New BC run directory, required when preparing")
    parser.add_argument("--device", choices=["mps", "cpu", "cuda", "auto"])
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--deadline", help="Optional earlier absolute deadline for the new run")
    args = parser.parse_args(argv)
    if args.prepare_from_search:
        if args.output is None:
            parser.error("--output is required with --prepare-from-search")
        ref = prepare_from_search(args.prepare_from_search, args.registration, args.output,
                                  device=args.device or "mps", max_hours=args.max_hours if args.max_hours is not None else 0.75,
                                  deadline=args.deadline)
        print(json.dumps({"status": "registered_not_started", "registration": ref}))
        return 0
    if any(value is not None for value in (args.output, args.deadline, args.device, args.max_hours)):
        parser.error("Training uses the prepared registration; output/device/budget options are prepare-only")
    context = validate_registration(args.registration)
    engine = BC(context)
    previous = {}
    def request_stop(signum, frame):
        engine.signal_requested = "signal_" + signal.Signals(signum).name.lower()
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, request_stop)
    try:
        result = engine.run()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(json.dumps({key: result[key] for key in ("stage", "status", "epoch", "bc_optimizer_steps", "bc_minibatch_samples",
          "replay_env_step_calls", "ppo_decisions", "network_policy_validated")}, ensure_ascii=False))
    return 0 if result["status"] == "dataset_fit_criterion_met" else 1 if result["status"] in {"error", "evidence_mismatch"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
