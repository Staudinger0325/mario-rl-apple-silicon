"""Load the locally trained World 1 release models; never download weights."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import torch

SCHEMA = "world1-local-state-dicts-v1"
METHODS = {"1-1": "ppo", "1-2": "ppo", "1-3": "own_search_bc", "1-4": "ppo"}


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tensor_digest(state):
    """Hash tensor names, dtypes, shapes and exact CPU bytes in sorted order."""
    if not isinstance(state, dict) or not state:
        raise ValueError("Release weights must be a nonempty plain state_dict")
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Release weights may contain tensors only")
        value = value.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError("Release weights contain non-finite values")
        result.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def default_manifest_path():
    directory = os.environ.get("MARIO_WEIGHTS_DIR")
    return (Path(directory).expanduser() if directory else Path(__file__).resolve().parents[1] / "weights") / "manifest.json"


def load_manifest(manifest_path=None):
    path = Path(manifest_path or default_manifest_path()).resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != SCHEMA or manifest.get("observation_shape") != [4, 84, 84]
            or manifest.get("action_set") != "simple7" or manifest.get("reward_mode") != "full"
            or manifest.get("castle_path_rules") is not False):
        raise ValueError("Expected the World 1 local state-dict release manifest")
    records = manifest.get("models", [])
    if len(records) != 4 or {r.get("level") for r in records} != set(METHODS):
        raise ValueError("The manifest must identify four distinct World 1 models")
    filenames = set()
    for row in records:
        level = row["level"]
        if (type(row.get("world")) is not int or type(row.get("stage")) is not int
                or level != f'{row["world"]}-{row["stage"]}' or row.get("method") != METHODS[level]):
            raise ValueError("Manifest level or training-method identity mismatch: " + level)
        name = Path(row["path"])
        if name.is_absolute() or ".." in name.parts or "\\" in row["path"] or row["path"] in ("", "."):
            raise ValueError("Model paths must be relative to the manifest")
        if row["path"].casefold() in filenames:
            raise ValueError("Duplicate model file path")
        filenames.add(row["path"].casefold())
        validation = row.get("validation", {})
        if (validation.get("wins") != 5 or validation.get("episodes") != 5
                or validation.get("fixed_start_map") is not True or validation.get("device") != "cpu"
                or validation.get("action_selection") != "argmax" or validation.get("seeds") != list(range(5))):
            raise ValueError("Missing historical fixed-start validation summary: " + level)
        if row["method"] == "own_search_bc" and row.get("selected_ppo_decisions") != 0:
            raise ValueError("The released BC policy must not be labeled as a PPO policy")
    return path, manifest


def resolve_weights(world=1, stage=1, manifest_path=None):
    """Return (path, record) after file/tensor checks. This is not a receipt audit."""
    if type(world) is not int or type(stage) is not int or (world, stage) not in {(1, s) for s in range(1, 5)}:
        raise ValueError("Bundled models cover World 1 stages 1-4 only; provide --checkpoint for another level")
    path, manifest = load_manifest(manifest_path)
    record = next(r for r in manifest["models"] if (r["world"], r["stage"]) == (world, stage))
    weight = path.parent / record["path"]
    if weight.is_symlink() or not weight.resolve().is_relative_to(path.parent):
        raise ValueError("Model path escapes the local weight directory")
    if file_digest(weight) != record["sha256"]:
        raise ValueError("Local model file checksum mismatch: " + record["level"])
    state = torch.load(weight, map_location="cpu", weights_only=True)
    if tensor_digest(state) != record["model_sha256"]:
        raise ValueError("Local model tensor checksum mismatch: " + record["level"])
    return weight, record


def ensure_weights(world=1, stage=1, manifest_path=None):
    """Compatibility helper returning verified bundled local weights, offline."""
    return resolve_weights(world, stage, manifest_path)[0]
