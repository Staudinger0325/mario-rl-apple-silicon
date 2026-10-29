#!/usr/bin/env python3
"""Independently validate one registered self-trajectory BC actor, never train.

Each action comes only from CPU argmax on the current image stack. The reference
trajectory is used after the first episode for a diagnostic, never for control.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCOPE = "own_trajectory_bc_closed_loop_validation"
SOURCE_KIND = "own_search_fresh_random_bc"
ACTIONS = (("NOOP",), ("right",), ("right", "A"), ("right", "B"),
           ("right", "A", "B"), ("A",), ("left",))
PROTOCOL = {"world": 1, "stage": 3, "device": "cpu", "action_selection": "argmax",
            "episodes": 5, "seeds": [0, 1, 2, 3, 4], "max_steps": 3000,
            "action_repeat": 4, "reward_mode": "full", "castle_path_rules": False,
            "fresh_environment_per_episode": True, "fixed_start": True,
            "action_set": "simple7", "video_fps": 15}
REF_NAMES = {"checkpoint", "initial_checkpoint", "dataset", "bc_registration",
             "bc_status", "candidate", "search_manifest"}
FINAL_BC_STATES = {"dataset_fit_criterion_met", "epoch_budget_exhausted", "deadline_reached",
                   "stop_requested", "signal_sigterm", "signal_sigint", "keyboard_interrupt"}


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize(value):
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite evidence value")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported evidence type: {type(value).__name__}")


def encoded(value):
    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(json.dumps(normalize(value), ensure_ascii=False, indent=2,
                                    allow_nan=False).encode() + b"\n")
    temporary.replace(path)


def plain_path(path, *, must_exist=True):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f"Symlink evidence/output is not allowed: {path}")
    if must_exist and not path.is_file():
        raise ValueError(f"Expected an existing regular file: {path}")
    return path


def checked(ref):
    if not isinstance(ref, dict) or not {"path", "sha256"} <= set(ref):
        raise ValueError("Evidence reference requires path and sha256")
    path = plain_path(ref["path"])
    raw = path.read_bytes()
    if digest(raw) != ref["sha256"] or ("bytes" in ref and ref["bytes"] != len(raw)):
        raise ValueError(f"Evidence bytes/hash differ: {path}")
    return path, raw


def file_ref(path):
    path, raw = plain_path(path), Path(path).read_bytes()
    return {"path": str(path), "sha256": digest(raw), "bytes": len(raw)}


def source_names():
    return {"scripts/validate_own_trajectory_policy.py", "scripts/train_own_trajectory_bc.py",
            "scripts/explore_own_prefixes.py"} | {
        str(path.relative_to(ROOT)) for path in (ROOT / "mario_rl").rglob("*.py")}


def tensor_hash(state):
    import torch
    result = hashlib.sha256()
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing model state")
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError("Non-finite or non-tensor model state")
        value = tensor.detach().cpu().contiguous()
        result.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def safe_load(raw):
    import torch
    result = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    if not isinstance(result, dict):
        raise ValueError("Expected a tensor-only checkpoint dictionary")
    return result


def require_same_ref(ref, captured, label):
    if (not isinstance(ref, dict) or ref.get("sha256") != digest(captured[1])
            or Path(ref.get("path", "")).absolute() != captured[0]):
        raise ValueError(f"BC checkpoint/manifest does not bind {label}")


def observation_fingerprint(obs):
    obs = np.asarray(obs)
    if (obs.shape != (4, 84, 84) or obs.dtype != np.float32
            or not np.isfinite(obs).all() or np.any(obs < 0) or np.any(obs > 1)):
        raise ValueError("Expected finite float32 (4,84,84) observations in [0,1]")
    return {"sha256": digest(obs.tobytes(order="C")), "shape": list(obs.shape), "dtype": obs.dtype.str}


def prepare_from_bc(bc_directory, registration, output, *, max_hours=0.25, deadline=None,
                    environment_kind="real_nes"):
    """Bind a completed portable BC run to a fresh independent policy evaluation."""
    from scripts import train_own_trajectory_bc as bc
    from scripts import explore_own_prefixes as search
    folder = plain_path(Path(bc_directory) / "final.pt").parent
    checkpoint = safe_load((folder / "final.pt").read_bytes())
    bc_reg_path, bc_reg_raw = checked(checkpoint.get("registration_ref"))
    bc_reg = json.loads(bc_reg_raw)
    if bc_reg.get("execution_mode") != "portable":
        raise ValueError("Prepare expects a portable BC run from its own registered search")
    refs = {"checkpoint": file_ref(folder / "final.pt"),
            "initial_checkpoint": file_ref(folder / "initial.pt"),
            "dataset": file_ref(folder / "dataset.json"),
            "bc_registration": file_ref(bc_reg_path),
            "bc_status": file_ref(folder / "manifest.json"),
            "candidate": file_ref(checkpoint["search_refs"]["candidate"]["path"]),
            "search_manifest": file_ref(checkpoint["search_refs"]["manifest"]["path"])}
    reg = {"schema_version": 1, "scope": SCOPE, "execution_mode": "portable",
           "created_at": now(), "historical_reproduction_claim": False,
           "refs": refs, "protocol": PROTOCOL, "output": str(Path(output).absolute()),
           "deadline": search.portable_deadline(max_hours, deadline),
           "checkpoint_source": {"kind": SOURCE_KIND, "model_sha256": checkpoint["model_sha256"],
                                 "initial_model_sha256": checkpoint["initial_model_sha256"]},
           "sources": {name: digest(plain_path(ROOT / name).read_bytes()) for name in source_names()}}
    return bc.write_new_registration(registration, reg,
        lambda path: audit_registration(path, environment_kind=environment_kind))


def audit_registration(path, *, environment_kind="real_nes"):
    """Read and bind actual model/data/source bytes. Does not construct a NES env."""
    if environment_kind not in {"real_nes", "synthetic_fixture"}:
        raise ValueError("Explicit supported environment kind required")
    import torch
    from mario_rl.model import ActorCritic
    torch.set_num_threads(1)
    reg_path = plain_path(path)
    reg_raw = reg_path.read_bytes()
    reg = json.loads(reg_raw)
    if reg.get("schema_version") != 1 or reg.get("scope") != SCOPE:
        raise ValueError("Expected own-trajectory BC validation schema/version")
    if reg.get("protocol") != PROTOCOL:
        raise ValueError("Validation protocol must exactly match the five-episode CPU protocol")
    deadline = datetime.fromisoformat(reg["deadline"])
    if deadline.utcoffset() is None or not math.isfinite(deadline.timestamp()):
        raise ValueError("A finite absolute deadline with timezone is required")
    if set(reg.get("refs", {})) != REF_NAMES:
        raise ValueError("Validation must pin all BC/search/input references")
    sources = reg.get("sources", {})
    if set(sources) != source_names():
        raise ValueError("Validation source registration is incomplete")
    source_snapshot = {}
    for name, expected in sources.items():
        raw = plain_path(ROOT / name).read_bytes()
        if digest(raw) != expected:
            raise ValueError(f"Current source differs from registered source: {name}")
        source_snapshot[name] = {"sha256": expected, "text": raw.decode("utf-8")}
    captured = {name: checked(ref) for name, ref in reg["refs"].items()}
    checkpoint, initial = (safe_load(captured[name][1]) for name in ("checkpoint", "initial_checkpoint"))
    status = json.loads(captured["bc_status"][1])
    bc_reg = json.loads(captured["bc_registration"][1])
    dataset = json.loads(captured["dataset"][1])
    candidate = json.loads(captured["candidate"][1])
    manifest = json.loads(captured["search_manifest"][1])
    from scripts import train_own_trajectory_bc as bc
    recipe, portable = bc.registration_recipe(bc_reg, environment_kind)
    if reg.get("execution_mode") not in (None, "portable") or portable != (reg.get("execution_mode") == "portable"):
        raise ValueError("Validation and BC execution modes differ")
    n = bc_reg.get("trajectory_steps")
    if type(n) is not int or n <= 0 or (environment_kind == "real_nes" and not portable and n != 439):
        raise ValueError("Unexpected registered self-trajectory length")
    if captured["checkpoint"][0].name != "final.pt":
        raise ValueError("Register the completed BC final.pt, not a mutable training checkpoint")
    origin = checkpoint.get("lineage", {}).get("initialization", {})
    expected_origin = {"kind": "scratch", "seed": recipe["seed"], "source_path": None, "source_sha256": None}
    if origin != expected_origin or initial.get("lineage", {}).get("initialization") != expected_origin:
        raise ValueError("Expected independently initialized registered-seed scratch origin")
    expected_config = {"num_inputs": 4, "num_actions": 7, "observation_shape": [4, 84, 84],
                       "world": 1, "stage": 3, "seed": recipe["seed"], "env_seed": recipe["env_seed"],
                       "reward_mode": "full", "castle_path_rules": False, "action_repeat": 4,
                       "max_environment_steps": manifest.get("config", {}).get("max_prefix")}
    limit = expected_config["max_environment_steps"]
    if (type(limit) is not int or limit < n
            or (environment_kind == "real_nes" and not portable and limit != 3000)):
        raise ValueError("Unexpected search/data environment time limit")
    for label, payload in (("checkpoint", checkpoint), ("initial", initial), ("status", status)):
        if (payload.get("stage") != "own_trajectory_bc" or payload.get("environment_kind") != environment_kind
                or payload.get("error") is not None or payload.get("ppo_updates") != 0
                or payload.get("ppo_decisions") != 0 or payload.get("network_policy_validated") is not False):
            raise ValueError(f"{label} is not error-free, real, BC-only evidence")
        if payload.get("model_config") != expected_config:
            raise ValueError(f"{label} model/environment contract differs")
        require_same_ref(payload.get("registration_ref"), captured["bc_registration"], "BC registration")
        require_same_ref(payload.get("dataset_ref"), captured["dataset"], "dataset")
        if payload.get("recipe") != bc_reg.get("recipe"):
            raise ValueError("BC recipe differs across source artifacts")
    if (checkpoint.get("status") not in FINAL_BC_STATES or checkpoint.get("status") != status.get("status")
            or type(checkpoint.get("bc_optimizer_steps")) is not int or checkpoint["bc_optimizer_steps"] <= 0):
        raise ValueError("BC has no completed finite training artifact")
    for key in ("epoch", "bc_optimizer_steps", "bc_minibatch_samples", "initial_model_sha256",
                "replay_env_step_calls", "matched_transitions", "returned_env_steps"):
        if checkpoint.get(key) != status.get(key):
            raise ValueError(f"BC final checkpoint and final status disagree: {key}")
    if (initial.get("bc_optimizer_steps") != 0 or initial.get("epoch") != 0
            or initial.get("optimizer", {}).get("state") != {}):
        raise ValueError("Initial model must precede all optimizer updates")
    model_hash, initial_hash = tensor_hash(checkpoint["model"]), tensor_hash(initial["model"])
    declared = reg.get("checkpoint_source", {})
    if declared != {"kind": SOURCE_KIND, "model_sha256": model_hash, "initial_model_sha256": initial_hash}:
        raise ValueError("Registered BC model/source hashes differ")
    if model_hash == initial_hash:
        raise ValueError("The registered model was not changed by BC training")
    for payload, expected_hash in ((checkpoint, model_hash), (initial, initial_hash)):
        if (payload.get("model_sha256") != expected_hash or payload.get("initial_model_sha256") != initial_hash
                or payload.get("lineage", {}).get("root_initial_model_sha256") != initial_hash
                or payload.get("lineage", {}).get("stages") != ["fresh_scratch", "own_trajectory_bc"]):
            raise ValueError("BC model hash/initial lineage is inconsistent")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(recipe["seed"])
        fresh = ActorCritic().cpu()
    if tensor_hash(fresh.state_dict()) != initial_hash:
        raise ValueError("Initial tensor bytes do not match an independent registered-seed fresh construction")
    fresh.load_state_dict(checkpoint["model"], strict=True)
    for name in ("critic_linear.weight", "critic_linear.bias"):
        if not torch.equal(initial["model"][name], checkpoint["model"][name]):
            raise ValueError("BC critic head was modified despite frozen-head provenance")
    for payload in (checkpoint, status):
        require_same_ref(payload.get("initial_checkpoint_ref"), captured["initial_checkpoint"], "initial checkpoint")
    for payload in (checkpoint, initial):
        for name in ("registration_ref", "dataset_ref", "initial_checkpoint_ref", "source_snapshot_ref", "search_refs"):
            if payload.get("lineage", {}).get(name) != payload.get(name):
                raise ValueError(f"BC lineage and top-level source references disagree: {name}")
    if bc_reg.get("scope") != "own_trajectory_bc" or bc_reg.get("trajectory_steps") != n:
        raise ValueError("BC registration does not describe the audited own trajectory")
    bc_sources = {name: value for name, value in sources.items() if name != "scripts/validate_own_trajectory_policy.py"}
    if bc_reg.get("sources") != bc_sources:
        raise ValueError("BC training and evaluation use different source/preprocessing implementations")
    captured["bc_source_snapshot"] = checked(checkpoint["source_snapshot_ref"])
    for payload in (initial, status):
        require_same_ref(payload.get("source_snapshot_ref"), captured["bc_source_snapshot"], "BC source snapshot")
    recorded_sources = json.loads(captured["bc_source_snapshot"][1])
    if set(recorded_sources) != set(bc_sources):
        raise ValueError("BC source snapshot is incomplete")
    for name, record in recorded_sources.items():
        if record.get("sha256") != bc_sources[name] or digest(record["text"].encode()) != bc_sources[name]:
            raise ValueError("BC source snapshot bytes differ")
    # Bind the original search artifacts and the BC-frozen copies without
    # pretending their intentionally different paths are the same source file.
    for key in ("manifest", "candidate", "verification", "source_snapshot"):
        original_ref = bc_reg.get("search", {}).get(key)
        copied_ref = checkpoint.get("search_refs", {}).get(key)
        captured["bc_search_" + key] = checked(copied_ref)
        raw = captured["bc_search_" + key][1]
        if not isinstance(original_ref, dict) or original_ref.get("sha256") != digest(raw):
            raise ValueError("BC changed the registered search source bytes")
        for payload in (initial, status, dataset):
            require_same_ref(payload.get("search_refs", {}).get(key), captured["bc_search_" + key], "search " + key)
    if (captured["bc_search_manifest"][1] != captured["search_manifest"][1]
            or captured["bc_search_candidate"][1] != captured["candidate"][1]):
        raise ValueError("The registered evaluation trajectory differs from BC's own source")
    for key in ("candidate", "verification", "source_snapshot"):
        name = Path(bc_reg["search"][key]["path"]).name
        raw = captured["bc_search_" + key][1]
        if manifest.get("artifacts", {}).get(name) != {"sha256": digest(raw), "bytes": len(raw)}:
            raise ValueError("Search manifest does not bind its actual candidate/verification/source")
    if (manifest.get("status") != "trajectory_verified" or manifest.get("model_created") is not False
            or manifest.get("environment_kind") != environment_kind
            or manifest.get("config", {}).get("seed") != recipe["seed"]
            or manifest.get("config", {}).get("env_seed") != recipe["env_seed"]):
        raise ValueError("Source is not model-free verified self exploration")
    if (candidate.get("scope") != "self_discovered_action_trajectory_not_network"
            or candidate.get("environment_kind") != environment_kind or candidate.get("reset_seed") != recipe["env_seed"]
            or len(candidate.get("steps", [])) != n or candidate.get("terminal_clear") is not True):
        raise ValueError("Incorrect own-trajectory identity")
    verified = [json.loads(line) for line in captured["bc_search_verification"][1].splitlines()]
    if len(verified) != n + 1 or verified[0] != {"kind": "reset", "snapshot": candidate.get("reset")}:
        raise ValueError("Incomplete original trajectory cold verification")
    for key in ("observations", "actions", "replay_trace"):
        captured["dataset_" + key] = checked(dataset[key])
    observations = np.load(io.BytesIO(captured["dataset_observations"][1]), allow_pickle=False)
    actions = np.load(io.BytesIO(captured["dataset_actions"][1]), allow_pickle=False)
    replay = [json.loads(line) for line in captured["dataset_replay_trace"][1].splitlines()]
    if (dataset.get("rows") != n or dataset.get("actual_terminal_clear") is not True
            or dataset.get("environment_kind") != environment_kind or dataset.get("matched_transitions") != n
            or dataset.get("replay_env_step_calls") != n or actions.dtype != np.int64
            or actions.shape != (n,) or observations.shape != (n, 4, 84, 84)
            or len(replay) != n + 1 or replay[0] != verified[0]):
        raise ValueError("Dataset/replay is incomplete or has different semantics")
    before = candidate["reset"]
    for index, (item, original_replay, dataset_replay) in enumerate(zip(candidate["steps"], verified[1:], replay[1:])):
        transition = item.get("transition", {})
        if (transition.get("before") != before or type(transition.get("action")) is not int
                or not 0 <= transition["action"] < 7 or int(actions[index]) != transition["action"]
                or observation_fingerprint(observations[index]) != transition["before"]["observation"]
                or item.get("transition_sha256") != digest(encoded(transition))
                or original_replay.get("transition") != transition
                or original_replay.get("transition_sha256") != item["transition_sha256"]
                or dataset_replay.get("transition") != transition or dataset_replay.get("matched") is not True
                or dataset_replay.get("step") != index + 1
                or dataset_replay.get("transition_sha256") != item["transition_sha256"]):
            raise ValueError("Dataset labels/inputs differ from the verified self trajectory")
        if index < n - 1 and (transition.get("terminated") or transition.get("truncated")):
            raise ValueError("Trajectory ended before its final decision")
        before = transition["after"]
    final = candidate["steps"][-1]["transition"]
    if not genuine_clear(final["after"]["info"], final.get("terminated"), final.get("truncated")):
        raise ValueError("Self trajectory has no authentic terminal clear")
    output = plain_path(reg["output"], must_exist=False)
    if output.exists():
        raise ValueError("Evaluation output must be a new directory; previous attempts are immutable")
    for name in ("checkpoint", "candidate", "initial_checkpoint"):
        parent = captured[name][0].parent
        if output == parent or output in parent.parents or parent in output.parents:
            raise ValueError("Keep validation output separate from BC/search source evidence")
    if reg.get("stop_file"):
        plain_path(reg["stop_file"], must_exist=False)
    return {"registration": reg, "registration_path": reg_path, "registration_raw": reg_raw,
            "captured": captured, "sources": source_snapshot, "candidate": candidate,
            "checkpoint": checkpoint, "initial": initial, "bc_status": status,
            "output": output, "deadline": deadline.timestamp(), "environment_kind": environment_kind}


def genuine_clear(info, terminated, truncated):
    flag = info.get("flag_get")
    return ((flag is True or type(flag) is int and flag == 1) and terminated is True
            and truncated is False and not info.get("TimeLimit.truncated", False)
            and info.get("castle_path_rules") is False
            and info.get("castle_path_rule_triggered", False) is False)


def checked_info(info):
    info = normalize(info)
    if (not isinstance(info, dict) or info.get("world") != 1 or info.get("stage") != 3
            or info.get("reward_mode") != "full" or info.get("castle_path_rules") is not False
            or info.get("castle_path_rule_triggered", False) is not False
            or info.get("reward_castle_rule", 0) != 0):
        raise ValueError("Evaluation environment changed level/reward/rules identity")
    return info


class Halt(Exception):
    pass


def inspect_video(path, expected_frames, resolution):
    """Decode/count the stream independently of append_data call counts."""
    import imageio.v2 as imageio
    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        frames = reader.count_frames()
    finally:
        reader.close()
    fps = float(metadata.get("fps", 0))
    if (type(expected_frames) is not int or expected_frames <= 0 or frames != expected_frames
            or not math.isfinite(fps) or abs(fps - 15) > 1e-5
            or tuple(metadata.get("size", ())) != tuple(resolution)):
        raise ValueError("Encoded video frames/fps/resolution differ from actual decisions")
    duration = frames / fps
    container = metadata.get("duration")
    if container is not None and (not math.isfinite(float(container))
            or abs(float(container) - duration) > max(1 / fps, 0.05)):
        raise ValueError("Encoded duration differs from frame count/fps")
    return {"frames": int(frames), "fps": fps, "resolution": list(resolution),
            "duration_seconds": duration, "container_duration_seconds": container}


def video_writer(path):
    import imageio.v2 as imageio
    return imageio.get_writer(path, fps=15, codec="libx264", macro_block_size=1,
                             output_params=["-threads", "2", "-movflags", "+faststart"])


def first_divergence(trace_path, candidate):
    """Offline only. No result of this function can influence environment actions."""
    rows = [json.loads(line) for line in Path(trace_path).read_text().splitlines() if line.strip()]
    records = [row for row in rows if row.get("kind") == "step"]
    first_obs = first_action = None
    for row, expected in zip(records, candidate["steps"]):
        transition = expected["transition"]
        if first_obs is None and row["before_observation"] != transition["before"]["observation"]:
            first_obs = {"step": row["step"], "actual": row["before_observation"],
                         "reference": transition["before"]["observation"]}
        if first_action is None and row["action"] != transition["action"]:
            first_action = {"step": row["step"], "actual": row["action"], "reference": transition["action"]}
    return {"performed_after_episode": True, "used_for_action_selection": False,
            "compared_steps": min(len(records), len(candidate["steps"])),
            "first_observation_divergence": first_obs, "first_action_divergence": first_action,
            "actual_steps": len(records), "reference_steps": len(candidate["steps"])}


class Validator:
    def __init__(self, context, *, env_factory=None, model=None, writer_factory=None,
                 video_inspector=None, clock=time.time):
        self.context, self.clock = context, clock
        self.real = context["environment_kind"] == "real_nes"
        if self.real and any(value is not None for value in (env_factory, model, writer_factory, video_inspector)):
            raise ValueError("Injected implementations are allowed only for explicit synthetic fixtures")
        self.reg, self.output = context["registration"], context["output"]
        self.env_factory, self.model = env_factory, model
        self.writer_factory, self.inspector = writer_factory or video_writer, video_inspector or inspect_video
        self.signal_requested = None
        self.issued = self.returned = self.resets = 0
        self.episodes, self.frozen = [], {}
        self.status, self.error = "initializing", None
        self.started = clock()
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "registration.json").write_bytes(context["registration_raw"])
        folder = self.output / "evidence"
        folder.mkdir()
        for name, (source, raw) in context["captured"].items():
            destination = folder / (name + Path(source).suffix)
            destination.write_bytes(raw)
            self.frozen[name] = {**file_ref(destination), "source_path": str(source)}
        write_json(self.output / "source-snapshot.json", context["sources"])
        self.persist()

    def check(self):
        if self.signal_requested:
            raise Halt(self.signal_requested)
        if (self.output / "STOP").exists() or (self.reg.get("stop_file") and Path(self.reg["stop_file"]).exists()):
            raise Halt("stop_requested")
        if self.clock() >= self.context["deadline"]:
            raise Halt("deadline_reached")

    def state(self):
        all_clear = len(self.episodes) == 5 and all(e.get("clear") is True and e.get("status") == "complete" for e in self.episodes)
        validated = self.real and self.status == "validated_5_of_5" and all_clear
        return {"schema_version": 1, "scope": SCOPE, "status": self.status, "recorded_at": now(),
                "environment_kind": self.context["environment_kind"], "checkpoint_source": self.reg["checkpoint_source"],
                "registration": file_ref(self.output / "registration.json"), "evidence": self.frozen,
                "source_snapshot": file_ref(self.output / "source-snapshot.json"), "protocol": PROTOCOL,
                "episodes": self.episodes, "complete_episodes": sum(e.get("status") == "complete" for e in self.episodes),
                "clear_episodes": sum(e.get("clear") is True for e in self.episodes),
                "reset_calls": self.resets, "issued_env_step_calls": self.issued,
                "returned_env_step_calls": self.returned, "unreturned_env_step_calls": self.issued - self.returned,
                "ppo_decisions": 0, "ppo_updates": 0, "bc_training_performed_by_validator": False,
                "network_policy_validated": validated, "world1_complete": False, "all32_complete": False,
                "project_complete": False, "error": self.error,
                "elapsed_seconds": self.clock() - self.started, "deadline": self.reg["deadline"],
                "notes": ["Five seeds repeat the fixed 1-3 map/start; not random-map generalization.",
                          "All actions are current-image CPU argmax; no action script, replay, or fallback.",
                          "Video is reset plus every decision frame, not all internal NES frames.",
                          "Deadline/STOP checks are cooperative around each env call; a blocked step needs external supervision."]}

    def persist(self):
        write_json(self.output / "status.json", self.state())

    def episode(self, seed):
        import torch
        self.check()
        directory = self.output / f"episode-{seed + 1}"
        directory.mkdir(exist_ok=False)
        trace_path, video_path = directory / "trace.jsonl", directory / "episode.mp4"
        trace = writer = env = None
        record = {"episode": seed + 1, "seed": seed, "status": "in_progress", "steps": 0,
                  "issued_env_step_calls": 0, "clear": False, "raw_return": 0.0, "shaped_return": 0.0}
        self.episodes.append(record)
        frames = 0
        resolution = None
        try:
            self.check()
            if self.env_factory is None:
                from mario_rl.env import make_env
                env = make_env(world=1, stage=3, seed=seed, max_steps=5000,
                               reward_mode="full", castle_path_rules=False)
            else:
                env = self.env_factory(seed)
            if env.action_space.n != 7 or tuple(map(tuple, env.actions)) != ACTIONS:
                raise ValueError("Unexpected action mapping")
            self.check()
            self.resets += 1
            obs, info = env.reset(seed=seed)
            info = checked_info(info)
            before = observation_fingerprint(obs)
            trace = trace_path.open("x", buffering=1)
            trace.write(encoded({"kind": "reset", "step": 0, "seed": seed,
                                 "observation": before, "info": info}).decode() + "\n")
            frame = np.asarray(env.render())
            if frame.shape != (240, 256, 3) or frame.dtype != np.uint8:
                raise ValueError("Expected uncropped native RGB video frames")
            resolution = (frame.shape[1], frame.shape[0])
            writer = self.writer_factory(video_path)
            writer.append_data(frame)
            frames += 1
            record["max_x"] = int(info.get("x_pos", 0))
            record["reset_observation"] = before
            for step in range(1, PROTOCOL["max_steps"] + 1):
                self.check()
                # The reference trajectory is deliberately unavailable here.
                with torch.inference_mode():
                    logits, _ = self.model(torch.as_tensor(obs).unsqueeze(0))
                    if tuple(logits.shape) != (1, 7) or not torch.isfinite(logits).all():
                        raise ValueError("Non-finite or invalid actor logits")
                    values = logits[0].cpu().tolist()
                    action = int(logits.argmax(dim=-1).item())
                self.check()
                self.issued += 1
                record["issued_env_step_calls"] += 1
                # Flush a write-ahead record, including calls that may raise.
                trace.write(encoded({"kind": "action_issued", "step": step, "action": action,
                    "before_observation": before, "logits": values,
                    "action_source": "cpu_argmax_current_observation"}).decode() + "\n")
                obs, reward, terminated, truncated, info = env.step(action)
                self.returned += 1
                if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
                    raise ValueError("Termination flags must be booleans")
                terminated, truncated = bool(terminated), bool(truncated)
                info = checked_info(info)
                after = observation_fingerprint(obs)
                reward, raw = float(reward), float(info["raw_reward"])
                if not math.isfinite(reward) or not math.isfinite(raw):
                    raise ValueError("Non-finite reward")
                row = {"kind": "step", "step": step, "before_observation": before,
                       "observation": after, "logits": values, "action": action,
                       "action_source": "cpu_argmax_current_observation", "reward": reward,
                       "raw_reward": raw, "terminated": terminated, "truncated": truncated, "info": info}
                trace.write(encoded(row).decode() + "\n")
                record.update(steps=step, final_info=info, terminated=terminated, truncated=truncated)
                record["raw_return"] += raw
                record["shaped_return"] += reward
                record["max_x"] = max(record["max_x"], int(info.get("x_pos", 0)))
                frame = np.asarray(env.render())
                if frame.shape != (240, 256, 3) or frame.dtype != np.uint8:
                    raise ValueError("Video frame format changed")
                writer.append_data(frame)
                frames += 1
                before = after
                self.check()  # A last step crossing STOP/deadline is not accepted.
                if terminated or truncated:
                    break
            record["evaluation_limit"] = not (terminated or truncated)
            record["clear"] = genuine_clear(info, terminated, truncated)
            record["end_reason"] = ("flag_get" if record["clear"] else "environment_truncation" if truncated
                                    else "terminal_without_clear" if terminated else "evaluation_limit")
            record["status"] = "complete"
        except Halt as error:
            record.update(status="interrupted", clear=False, interruption=str(error))
            raise
        except BaseException as error:
            record.update(status="error", clear=False, error={"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            cleanup_errors = []
            for resource in (writer, trace, env):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as error:
                        cleanup_errors.append(str(error))
            if trace_path.exists():
                record["trace"] = file_ref(trace_path)
            if video_path.exists():
                record["video"] = file_ref(video_path)
                try:
                    record["encoded_video"] = self.inspector(video_path, frames, resolution)
                except Exception as error:
                    cleanup_errors.append("video integrity: " + str(error))
            elif frames:
                cleanup_errors.append("Recorded video file is missing")
            record["video_frames_appended"] = frames
            if record["status"] == "complete" and frames != record["steps"] + 1:
                cleanup_errors.append("Video must contain reset plus every decision")
            if cleanup_errors:
                record.update(status="error", clear=False, artifact_errors=cleanup_errors)
            write_json(directory / "result.json", record)
            record["result"] = file_ref(directory / "result.json")
            self.persist()
            if cleanup_errors:
                raise ValueError("; ".join(cleanup_errors))
        return record

    def run(self):
        try:
            self.check()
            import torch
            torch.set_num_threads(1)
            if self.model is None:
                from mario_rl.model import load_model
                self.model = load_model(self.frozen["checkpoint"]["path"], device="cpu")
                if tensor_hash(self.model.state_dict()) != self.reg["checkpoint_source"]["model_sha256"]:
                    raise ValueError("Loaded frozen policy differs from registered BC tensors")
            self.model.cpu().eval()
            self.status = "evaluating"
            self.persist()
            for seed in PROTOCOL["seeds"]:
                self.episode(seed)
                if seed == 0:
                    diagnostic = first_divergence(self.episodes[0]["trace"]["path"], self.context["candidate"])
                    write_json(self.output / "first-episode-reference-diagnostic.json", diagnostic)
                self.check()
            all_clear = all(item["clear"] and item["status"] == "complete" for item in self.episodes)
            self.status = ("validated_5_of_5" if self.real else "synthetic_fixture_passed") if all_clear else "failed"
        except Halt as error:
            self.status, self.error = "interrupted", {"type": "Halt", "message": str(error)}
        except (KeyboardInterrupt, Exception) as error:
            self.status, self.error = "error", {"type": type(error).__name__, "message": str(error)}
        finally:
            self.persist()
        return self.state()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true", help="Verify bytes/source/data without creating an environment or output")
    parser.add_argument("--prepare-from-bc", type=Path, help="Register completed portable BC evidence; do not run a game")
    parser.add_argument("--output", type=Path, help="New validation attempt directory, required when preparing")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--deadline", help="Optional earlier absolute deadline")
    args = parser.parse_args(argv)
    if args.prepare_from_bc:
        if args.output is None or args.audit_only:
            parser.error("Preparation requires --output and is separate from --audit-only")
        ref = prepare_from_bc(args.prepare_from_bc, args.registration, args.output,
                              max_hours=args.max_hours if args.max_hours is not None else 0.25, deadline=args.deadline)
        print(json.dumps({"status": "registered_not_started", "registration": ref}))
        return 0
    if any(value is not None for value in (args.output, args.deadline, args.max_hours)):
        parser.error("Evaluation uses the prepared registration; output/budget options are prepare-only")
    context = audit_registration(args.registration)
    if args.audit_only:
        print(json.dumps({"status": "audit_passed", "scope": SCOPE, "checkpoint_source": context["registration"]["checkpoint_source"],
                          "network_policy_validated": False, "environment_created": False}, ensure_ascii=False))
        return 0
    engine = Validator(context)
    handlers = {}
    def stop(signum, frame):
        engine.signal_requested = "signal_" + signal.Signals(signum).name.lower()
    for signum in (signal.SIGINT, signal.SIGTERM):
        handlers[signum] = signal.signal(signum, stop)
    try:
        result = engine.run()
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    print(json.dumps({key: result[key] for key in ("status", "clear_episodes", "issued_env_step_calls", "network_policy_validated",
                                                 "world1_complete", "all32_complete")}, ensure_ascii=False))
    return 0 if result["network_policy_validated"] else 1 if result["status"] == "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
