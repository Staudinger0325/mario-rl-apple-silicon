"""Synthetic environments and tiny networks only; never construct NES or use real trajectories."""
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from scripts import explore_own_prefixes as search
from scripts import train_own_trajectory_bc as bc
from tests.test_own_prefixes import SyntheticEnv, config as search_config


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_linear = nn.Linear(1, 7)
        self.critic_linear = nn.Linear(1, 1)
        nn.init.zeros_(self.actor_linear.weight)
        nn.init.zeros_(self.actor_linear.bias)
        with torch.no_grad():
            self.actor_linear.bias[0] = 1

    def forward(self, x):
        feature = x.mean((1, 2, 3)).unsqueeze(1)
        return self.actor_linear(feature), self.critic_linear(feature)


class ReplacingActor(TinyActor):
    """Simulate a conversion replacing Parameter objects, without needing MPS."""
    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        for module in self.modules():
            for name, parameter in list(module._parameters.items()):
                if parameter is not None:
                    module._parameters[name] = nn.Parameter(parameter.detach().clone(), requires_grad=parameter.requires_grad)
        return result


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fixture(folder):
    folder = Path(folder).resolve()
    engine = search.Search(search_config(folder / "search", seed=42), lambda: SyntheticEnv(length=3),
                           environment_kind="synthetic_fixture")
    with mock.patch.object(engine.rng, "randrange", return_value=0):
        result = engine.run()
    assert result["trajectory_verified"]
    names = {"manifest": "manifest.json", "candidate": "candidate-trajectory.json",
             "verification": "trajectory-verification.jsonl", "source_snapshot": "source-snapshot.json"}
    recipe = {**bc.RECIPE, "device": "cpu", "deadline": datetime.fromtimestamp(time.time() + 600, timezone.utc).isoformat()}
    reg = {"schema_version": 1, "scope": bc.STAGE, "output": str(folder / "bc"), "trajectory_steps": 3,
           "recipe": recipe, "sources": {name: search.file_hash(bc.ROOT / name) for name in bc.source_names()},
           "search": {key: {"path": str(engine.output / name), "sha256": search.file_hash(engine.output / name)}
                      for key, name in names.items()}}
    path = folder / "registration.json"
    write(path, reg)
    return path, reg


def resign_search(path, reg, candidate=None, verified=None):
    if candidate is not None:
        write(Path(reg["search"]["candidate"]["path"]), candidate)
    if verified is not None:
        Path(reg["search"]["verification"]["path"]).write_text("".join(json.dumps(row) + "\n" for row in verified))
    manifest_path = Path(reg["search"]["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    for key in ("candidate", "verification", "source_snapshot"):
        artifact = Path(reg["search"][key]["path"])
        reg["search"][key]["sha256"] = search.file_hash(artifact)
        manifest["artifacts"][artifact.name] = {"sha256": search.file_hash(artifact), "bytes": artifact.stat().st_size}
    write(manifest_path, manifest)
    reg["search"]["manifest"]["sha256"] = search.file_hash(manifest_path)
    write(path, reg)


class OwnTrajectoryBCTests(unittest.TestCase):
    def test_end_to_end_synthetic_dataset_fresh_weights_actor_only_and_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reg = fixture(directory)
            env = SyntheticEnv(length=3)
            # Neither own nor external old checkpoints can be loaded by training.
            with mock.patch.object(torch, "load", side_effect=AssertionError("No checkpoint input permitted")):
                result = bc.run_bc(path, environment_kind="synthetic_fixture", env_factory=lambda: env, model_factory=TinyActor)
            self.assertEqual(result["status"], "dataset_fit_criterion_met")
            self.assertEqual(result["epoch"], 5)
            self.assertEqual(result["consecutive_perfect_epochs"], 5)
            self.assertEqual(env.calls, 3)
            self.assertEqual(result["replay_env_step_calls"], 3)
            self.assertEqual(result["matched_transitions"], 3)
            self.assertEqual(result["ppo_decisions"], 0)
            self.assertEqual(result["bc_optimizer_steps"], 5)
            self.assertEqual(result["bc_minibatch_samples"], 15)
            self.assertFalse(result["network_policy_validated"])
            output = Path(reg["output"])
            x = np.load(output / "observations.npy", allow_pickle=False)
            y = np.load(output / "actions.npy", allow_pickle=False)
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(x.shape, (3, 4, 84, 84))
            np.testing.assert_array_equal(x[:, 0, 0, 0], np.array([0, .01, .02], dtype=np.float32))
            np.testing.assert_array_equal(y, [0, 0, 0])
            initial = torch.load(output / "initial.pt", weights_only=True)
            final = torch.load(output / "final.pt", weights_only=True)
            self.assertEqual(initial["optimizer"]["state"], {})
            self.assertEqual(initial["bc_optimizer_steps"], 0)
            self.assertEqual(final["stage"], "own_trajectory_bc")
            self.assertEqual(final["initial_checkpoint_ref"]["sha256"], search.file_hash(output / "initial.pt"))
            self.assertEqual(final["dataset_ref"]["sha256"], search.file_hash(output / "dataset.json"))
            self.assertEqual(final["lineage"]["initialization"]["source_path"], None)
            for name in ("critic_linear.weight", "critic_linear.bias"):
                self.assertTrue(torch.equal(initial["model"][name], final["model"][name]))
            self.assertFalse(torch.equal(initial["model"]["actor_linear.bias"], final["model"]["actor_linear.bias"]))
            self.assertEqual(final["model_config"]["stage"], 3)
            rows = list(csv.DictReader(io.StringIO((output / "metrics.csv").read_text())))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row["cpu_accuracy"] == "1.0" and row["completed_epoch"] == "True" for row in rows))

    def test_optimizer_is_bound_after_device_replaces_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reg = fixture(directory)
            engine = bc.BC(bc.validate_registration(path, environment_kind="synthetic_fixture"),
                           env_factory=lambda: SyntheticEnv(length=3), model_factory=ReplacingActor)
            x, y = engine.collect()
            engine.initialize()
            actual = [p for group in engine.optimizer.param_groups for p in group["params"]]
            expected = [p for p in engine.model.parameters() if p.requires_grad]
            self.assertEqual([id(p) for p in actual], [id(p) for p in expected])
            self.assertEqual(engine.optimizer.state, {})
            before = engine.model.actor_linear.bias.detach().clone()
            engine.fit(x, y)
            self.assertFalse(torch.equal(before, engine.model.actor_linear.bias))

    def test_hash_tamper_and_missing_reference_keys_refused_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reg = fixture(directory)
            candidate = Path(reg["search"]["candidate"]["path"])
            candidate.write_bytes(candidate.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Registered bytes"):
                bc.validate_registration(path, environment_kind="synthetic_fixture")
            self.assertFalse(Path(reg["output"]).exists())
            with self.assertRaisesRegex(ValueError, "path and sha256"):
                bc.checked({"path": str(candidate), "bytes": 0})

    def test_recipe_source_and_wrong_environment_kind_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reg = fixture(directory)
            reg["recipe"]["lr"] = .01
            write(path, reg)
            with self.assertRaisesRegex(ValueError, "recipe change"):
                bc.validate_registration(path, environment_kind="synthetic_fixture")
            reg["recipe"]["lr"] = .001
            reg["sources"]["mario_rl/model.py"] = "0" * 64
            write(path, reg)
            with self.assertRaisesRegex(ValueError, "source differs"):
                bc.validate_registration(path, environment_kind="synthetic_fixture")
            reg["sources"]["mario_rl/model.py"] = search.file_hash(bc.ROOT / "mario_rl/model.py")
            write(path, reg)
            with self.assertRaises(ValueError):
                bc.validate_registration(path)  # A synthetic run cannot claim real NES.

    def test_existing_output_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reg = fixture(directory)
            Path(reg["output"]).mkdir()
            with self.assertRaisesRegex(ValueError, "new directory"):
                bc.validate_registration(path, environment_kind="synthetic_fixture")
            Path(reg["output"]).rmdir()
            link = Path(directory) / "alias.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "symlink"):
                bc.validate_registration(link, environment_kind="synthetic_fixture")

    def test_reset_and_step_mismatches_prevent_dataset_and_model(self):
        for reset in (True, False):
            with self.subTest(reset=reset), tempfile.TemporaryDirectory() as directory:
                path, reg = fixture(directory)
                if reset:
                    env = SyntheticEnv(length=3, on_reset=lambda env, obs, info: info.update(time=999))
                else:
                    env = SyntheticEnv(length=3, on_step=lambda env, obs, reward, terminated, truncated, info: (obs, reward + 1, terminated, truncated, info))
                result = bc.run_bc(path, environment_kind="synthetic_fixture", env_factory=lambda: env, model_factory=TinyActor)
                self.assertEqual(result["status"], "evidence_mismatch")
                self.assertEqual(result["replay_env_step_calls"], 0 if reset else 1)
                self.assertFalse((Path(reg["output"]) / "dataset.json").exists())
                self.assertFalse((Path(reg["output"]) / "initial.pt").exists())

    def test_conflicting_input_labels_rejected_even_with_resigned_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reg = fixture(directory)
            candidate = json.loads(Path(reg["search"]["candidate"]["path"]).read_text())
            steps = candidate["steps"]
            steps[0]["transition"]["after"]["observation"] = candidate["reset"]["observation"]
            steps[1]["transition"]["before"] = steps[0]["transition"]["after"]
            steps[1]["transition"]["action"] = 1
            for step in steps:
                step["transition_sha256"] = search.json_hash(step["transition"])
            verified = [{"kind": "reset", "snapshot": candidate["reset"]}] + steps
            resign_search(path, reg, candidate, verified)
            with self.assertRaisesRegex(ValueError, "conflicting"):
                bc.validate_registration(path, environment_kind="synthetic_fixture")

    def test_final_truncation_cannot_be_promoted_to_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reg = fixture(directory)
            candidate = json.loads(Path(reg["search"]["candidate"]["path"]).read_text())
            candidate["steps"][-1]["transition"]["truncated"] = True
            candidate["steps"][-1]["transition_sha256"] = search.json_hash(candidate["steps"][-1]["transition"])
            resign_search(path, reg, candidate, [{"kind": "reset", "snapshot": candidate["reset"]}] + candidate["steps"])
            with self.assertRaisesRegex(ValueError, "terminal clear"):
                bc.validate_registration(path, environment_kind="synthetic_fixture")

    def test_env_step_exception_is_charged_and_preserved(self):
        def fail(*args):
            raise RuntimeError("fixture advanced then failed")
        with tempfile.TemporaryDirectory() as directory:
            path, reg = fixture(directory)
            env = SyntheticEnv(length=3, on_step=fail)
            result = bc.run_bc(path, environment_kind="synthetic_fixture", env_factory=lambda: env, model_factory=TinyActor)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["replay_env_step_calls"], 1)
            self.assertEqual(result["returned_env_steps"], 0)
            self.assertEqual(result["unreturned_env_step_calls"], 1)
            self.assertIn("fixture advanced", result["error"]["message"])

    def test_stop_during_final_replay_does_not_promote_or_train(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reg = fixture(directory)
            def stop(env, *args):
                if env.calls == 3:
                    (Path(reg["output"]) / "STOP").touch()
            env = SyntheticEnv(length=3, on_step=stop)
            result = bc.run_bc(path, environment_kind="synthetic_fixture", env_factory=lambda: env, model_factory=TinyActor)
            self.assertEqual(result["status"], "stop_requested")
            self.assertEqual(result["matched_transitions"], 3)
            self.assertIsNone(result["dataset_ref"])
            self.assertFalse(result["dataset_fit_criterion_met"])

    def test_deadline_before_replay_creates_no_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reg = fixture(directory)
            reg["recipe"]["deadline"] = datetime.fromtimestamp(time.time() - 1, timezone.utc).isoformat()
            write(path, reg)
            factory = mock.Mock(side_effect=AssertionError("No env after deadline"))
            result = bc.run_bc(path, environment_kind="synthetic_fixture", env_factory=factory, model_factory=TinyActor)
            self.assertEqual(result["status"], "deadline_reached")
            self.assertEqual(result["replay_env_step_calls"], 0)
            factory.assert_not_called()

    def test_initialization_failure_is_not_masked_and_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reg = fixture(directory)
            calls = 0
            def factory():
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("independent reconstruction fixture failure")
                return TinyActor()
            result = bc.run_bc(path, environment_kind="synthetic_fixture", env_factory=lambda: SyntheticEnv(length=3), model_factory=factory)
            self.assertEqual(result["status"], "error")
            self.assertIn("reconstruction", result["error"]["message"])
            self.assertIsNotNone(result["dataset_ref"])
            self.assertFalse((Path(reg["output"]) / "final.pt").exists())

    def test_stop_mid_epoch_saves_exact_updates_and_incomplete_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reg = fixture(directory)
            original = torch.optim.Adam.step
            def stop(optimizer, *args, **kwargs):
                result = original(optimizer, *args, **kwargs)
                (Path(reg["output"]) / "STOP").touch()
                return result
            with mock.patch.object(torch.optim.Adam, "step", stop):
                result = bc.run_bc(path, environment_kind="synthetic_fixture", env_factory=lambda: SyntheticEnv(length=3), model_factory=TinyActor)
            self.assertEqual(result["status"], "stop_requested")
            self.assertEqual(result["bc_optimizer_steps"], 1)
            self.assertEqual(result["bc_minibatch_samples"], 3)
            self.assertFalse(result["dataset_fit_criterion_met"])
            final = torch.load(Path(reg["output"]) / "final.pt", weights_only=True)
            self.assertEqual(final["status"], "stop_requested")
            self.assertEqual(final["bc_optimizer_steps"], 1)
            rows = list(csv.DictReader(io.StringIO((Path(reg["output"]) / "metrics.csv").read_text())))
            self.assertEqual(rows[0]["completed_epoch"], "False")
            self.assertEqual(rows[0]["cpu_accuracy"], "")

    def test_close_error_still_saves_original_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reg = fixture(directory)
            env = SyntheticEnv(length=3, on_reset=lambda env, obs, info: info.update(time=999))
            env.close = mock.Mock(side_effect=RuntimeError("close also failed"))
            result = bc.run_bc(path, environment_kind="synthetic_fixture", env_factory=lambda: env, model_factory=TinyActor)
            self.assertEqual(result["status"], "evidence_mismatch")
            self.assertIn("Cold-reset", result["error"]["message"])
            self.assertTrue((Path(reg["output"]) / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
