"""Synthetic-only BC producer/consumer and closed-loop evidence regression tests."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from scripts import validate_own_trajectory_policy as validation
from scripts import train_own_trajectory_bc as bc
from tests.test_own_trajectory_bc import fixture as bc_fixture, TinyActor
from tests.test_own_prefixes import SyntheticEnv, Clock


class VideoEnv(SyntheticEnv):
    def render(self):
        return np.full((240, 256, 3), self._steps, dtype=np.uint8)


class Policy(nn.Module):
    def __init__(self, action=6, nonfinite=False):
        super().__init__()
        self.action, self.nonfinite = action, nonfinite
        self.seen = []

    def forward(self, x):
        self.seen.append(x.clone())
        logits = torch.zeros((len(x), 7))
        logits[:, self.action] = float("nan") if self.nonfinite else 1
        return logits, torch.zeros((len(x), 1))


class Writer:
    def __init__(self, path):
        self.path, self.frames = Path(path), 0

    def append_data(self, frame):
        assert frame.shape == (240, 256, 3)
        self.frames += 1

    def close(self):
        self.path.write_text(json.dumps({"frames": self.frames}))


def inspect(path, frames, resolution):
    actual = json.loads(Path(path).read_text())["frames"]
    if actual != frames:
        raise ValueError("Truncated encoded video")
    return {"frames": actual, "fps": 15, "resolution": list(resolution), "duration_seconds": actual / 15}


def fake_context(folder, *, deadline=10000):
    folder = Path(folder).resolve()
    env = VideoEnv(length=3)
    obs, _ = env.reset(seed=0)
    steps = []
    for _ in range(3):
        before = validation.observation_fingerprint(obs)
        obs, _, _, _, _ = env.step(0)
        steps.append({"transition": {"action": 0, "before": {"observation": before}}})
    reg = {"checkpoint_source": {"kind": validation.SOURCE_KIND, "model_sha256": "synthetic", "initial_model_sha256": "synthetic"},
           "deadline": datetime.fromtimestamp(deadline, timezone.utc).isoformat(), "output": str(folder / "validation")}
    return {"registration": reg, "registration_raw": validation.encoded(reg), "environment_kind": "synthetic_fixture",
            "output": folder / "validation", "deadline": deadline, "sources": {},
            "captured": {"checkpoint": (folder / "synthetic.pt", b"SYNTHETIC TEST ONLY")},
            "candidate": {"steps": steps}}


def produced_registration(folder):
    """Use actual BC producer output, avoiding separately invented consumer fields."""
    folder = Path(folder).resolve()
    path, source_reg = bc_fixture(folder)
    result = bc.run_bc(path, environment_kind="synthetic_fixture",
                       env_factory=lambda: SyntheticEnv(length=3), model_factory=TinyActor)
    assert result["status"] == "dataset_fit_criterion_met"
    output = Path(source_reg["output"])
    checkpoint = torch.load(output / "final.pt", weights_only=True)
    refs = {"checkpoint": validation.file_ref(output / "final.pt"),
            "initial_checkpoint": validation.file_ref(output / "initial.pt"),
            "dataset": validation.file_ref(output / "dataset.json"),
            "bc_registration": validation.file_ref(path),
            "bc_status": validation.file_ref(output / "manifest.json"),
            "candidate": validation.file_ref(Path(source_reg["search"]["candidate"]["path"])),
            "search_manifest": validation.file_ref(Path(source_reg["search"]["manifest"]["path"]))}
    reg = {"schema_version": 1, "scope": validation.SCOPE, "refs": refs,
           "checkpoint_source": {"kind": validation.SOURCE_KIND, "model_sha256": checkpoint["model_sha256"],
                                 "initial_model_sha256": checkpoint["initial_model_sha256"]},
           "protocol": validation.PROTOCOL, "deadline": source_reg["recipe"]["deadline"],
           "output": str(folder / "validation"),
           "sources": {name: validation.digest((validation.ROOT / name).read_bytes()) for name in validation.source_names()}}
    registration = folder / "evaluation-registration.json"
    validation.write_json(registration, reg)
    return registration, reg


class BCValidationTests(unittest.TestCase):
    def test_real_producer_to_consumer_evidence_schema_and_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            registration, reg = produced_registration(directory)
            with mock.patch("mario_rl.model.ActorCritic", TinyActor):
                context = validation.audit_registration(registration, environment_kind="synthetic_fixture")
            self.assertFalse(context["output"].exists())  # Audit does not run gameplay or create output.
            environments = []
            def factory(seed):
                env = VideoEnv(length=3)
                environments.append(env)
                return env
            actor = TinyActor()
            actor.load_state_dict(context["checkpoint"]["model"])
            engine = validation.Validator(context, model=actor, env_factory=factory,
                                          writer_factory=Writer, video_inspector=inspect)
            result = engine.run()
            self.assertEqual(result["status"], "synthetic_fixture_passed", result)
            self.assertEqual(result["clear_episodes"], 5)
            self.assertEqual(result["issued_env_step_calls"], 15)
            self.assertEqual(result["reset_calls"], 5)
            self.assertFalse(result["network_policy_validated"])
            self.assertFalse(result["world1_complete"])
            self.assertFalse(result["all32_complete"])
            self.assertEqual(len({id(e) for e in environments}), 5)
            self.assertEqual([e._seed for e in environments], [0, 1, 2, 3, 4])
            self.assertTrue(all(e._closed for e in environments))
            self.assertEqual(Path(result["evidence"]["checkpoint"]["path"]).read_bytes(), Path(reg["refs"]["checkpoint"]["path"]).read_bytes())
            for episode in result["episodes"]:
                self.assertEqual(episode["encoded_video"]["frames"], episode["steps"] + 1)
                self.assertEqual(episode["trace"]["sha256"], validation.digest(Path(episode["trace"]["path"]).read_bytes()))
                self.assertTrue(Path(episode["result"]["path"]).is_file())

    def test_model_actions_never_replaced_by_reference_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            context = fake_context(directory)
            environments = []
            def factory(seed):
                env = VideoEnv(length=3)
                environments.append(env)
                return env
            policy = Policy(action=6)
            result = validation.Validator(context, model=policy, env_factory=factory, writer_factory=Writer,
                                          video_inspector=inspect, clock=Clock()).run()
            self.assertEqual(result["clear_episodes"], 5)
            self.assertTrue(all(e.seen_actions == [6, 6, 6] for e in environments))
            self.assertEqual(len(policy.seen), 15)
            diagnostic = json.loads((context["output"] / "first-episode-reference-diagnostic.json").read_text())
            self.assertEqual(diagnostic["first_action_divergence"], {"step": 1, "actual": 6, "reference": 0})
            self.assertIsNone(diagnostic["first_observation_divergence"])
            self.assertFalse(diagnostic["used_for_action_selection"])

    def test_failure_is_preserved_for_all_five_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            context = fake_context(directory)
            result = validation.Validator(context, model=Policy(), env_factory=lambda seed: VideoEnv(length=2, clear=False),
                                          writer_factory=Writer, video_inspector=inspect, clock=Clock()).run()
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["complete_episodes"], 5)
            self.assertEqual(result["clear_episodes"], 0)
            for episode in result["episodes"]:
                self.assertEqual(episode["end_reason"], "terminal_without_clear")
                self.assertTrue(Path(episode["video"]["path"]).is_file())
                self.assertEqual(episode["encoded_video"]["frames"], 3)
            with self.assertRaises(FileExistsError):
                validation.Validator(context, model=Policy(), env_factory=lambda seed: VideoEnv())

    def test_flag_without_terminal_and_truncated_flag_are_not_wins(self):
        for truncated in (False, True):
            with self.subTest(truncated=truncated), tempfile.TemporaryDirectory() as directory:
                context = fake_context(directory)
                def hook(env, obs, reward, terminated, trunc, info):
                    info["flag_get"] = np.uint8(1)
                    return obs, reward, False, truncated, info
                factory = lambda seed: VideoEnv(length=100, on_step=hook)
                with mock.patch.dict(validation.PROTOCOL, {"max_steps": 2}):
                    result = validation.Validator(context, model=Policy(), env_factory=factory, writer_factory=Writer,
                                                  video_inspector=inspect, clock=Clock()).run()
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["clear_episodes"], 0)
                self.assertTrue(all(e["evaluation_limit"] is (not truncated) for e in result["episodes"]))

    def test_last_step_deadline_or_stop_cannot_promote_clear(self):
        for mode in ("deadline", "stop"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                clock = Clock()
                context = fake_context(directory, deadline=clock.value + 10)
                def hook(env, *args):
                    if mode == "deadline":
                        clock.value = context["deadline"]
                    else:
                        (context["output"] / "STOP").write_text("test stop")
                result = validation.Validator(context, model=Policy(), env_factory=lambda seed: VideoEnv(length=1, on_step=hook),
                                              writer_factory=Writer, video_inspector=inspect, clock=clock).run()
                self.assertEqual(result["status"], "interrupted", result)
                self.assertEqual(result["issued_env_step_calls"], 1)
                self.assertEqual(result["returned_env_step_calls"], 1)
                self.assertEqual(result["clear_episodes"], 0)
                self.assertTrue(result["episodes"][0]["final_info"]["flag_get"])
                self.assertEqual(result["episodes"][0]["encoded_video"]["frames"], 2)

    def test_preexisting_stop_does_not_create_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            context = fake_context(directory)
            stop = Path(directory) / "STOP"
            stop.write_text("stop")
            context["registration"]["stop_file"] = str(stop)
            factory = mock.Mock(side_effect=AssertionError("must not construct env"))
            result = validation.Validator(context, model=Policy(), env_factory=factory, writer_factory=Writer,
                                          video_inspector=inspect, clock=Clock()).run()
            self.assertEqual(result["status"], "interrupted")
            self.assertEqual(result["issued_env_step_calls"], 0)
            factory.assert_not_called()

    def test_issued_step_exception_is_charged_and_write_ahead_record_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            context = fake_context(directory)
            env = VideoEnv()
            env.step = mock.Mock(side_effect=RuntimeError("synthetic step failed after issue"))
            result = validation.Validator(context, model=Policy(), env_factory=lambda seed: env,
                                          writer_factory=Writer, video_inspector=inspect, clock=Clock()).run()
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["issued_env_step_calls"], 1)
            self.assertEqual(result["returned_env_step_calls"], 0)
            self.assertEqual(result["unreturned_env_step_calls"], 1)
            rows = [json.loads(line) for line in Path(result["episodes"][0]["trace"]["path"]).read_text().splitlines()]
            self.assertEqual([row["kind"] for row in rows], ["reset", "action_issued"])
            self.assertTrue(env._closed)

    def test_nonfinite_logits_are_rejected_before_step(self):
        with tempfile.TemporaryDirectory() as directory:
            context = fake_context(directory)
            env = VideoEnv()
            result = validation.Validator(context, model=Policy(nonfinite=True), env_factory=lambda seed: env,
                                          writer_factory=Writer, video_inspector=inspect, clock=Clock()).run()
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["issued_env_step_calls"], 0)
            self.assertEqual(env.calls, 0)

    def test_video_integrity_failure_cannot_pass_even_with_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            context = fake_context(directory)
            result = validation.Validator(context, model=Policy(), env_factory=lambda seed: VideoEnv(length=1),
                                          writer_factory=Writer, video_inspector=mock.Mock(side_effect=ValueError("missing encoded frame")),
                                          clock=Clock()).run()
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["clear_episodes"], 0)
            self.assertTrue(result["episodes"][0]["final_info"]["flag_get"])
            self.assertIn("missing encoded frame", result["episodes"][0]["artifact_errors"][0])

    def test_source_checkpoint_hash_and_synthetic_as_real_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            registration, reg = produced_registration(directory)
            with self.assertRaisesRegex(ValueError, "trajectory length|recipe change"):
                validation.audit_registration(registration)
            with mock.patch("mario_rl.model.ActorCritic", TinyActor):
                validation.audit_registration(registration, environment_kind="synthetic_fixture")
                Path(reg["refs"]["checkpoint"]["path"]).write_bytes(b"altered bytes")
                with self.assertRaisesRegex(ValueError, "bytes/hash"):
                    validation.audit_registration(registration, environment_kind="synthetic_fixture")
            self.assertFalse(Path(reg["output"]).exists())
            reg["sources"]["mario_rl/env.py"] = "0" * 64
            validation.write_json(registration, reg)
            with self.assertRaisesRegex(ValueError, "Current source"):
                validation.audit_registration(registration, environment_kind="synthetic_fixture")

    def test_error_status_and_non_scratch_initial_refused(self):
        for mode in ("error", "initial"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                registration, reg = produced_registration(directory)
                if mode == "error":
                    path = Path(reg["refs"]["bc_status"]["path"])
                    status = json.loads(path.read_text())
                    status["error"] = {"type": "TrainingError", "message": "test"}
                    validation.write_json(path, status)
                    reg["refs"]["bc_status"] = validation.file_ref(path)
                else:
                    path = Path(reg["refs"]["initial_checkpoint"]["path"])
                    checkpoint = torch.load(path, weights_only=True)
                    checkpoint["lineage"]["initialization"]["source_path"] = "old-weights.pt"
                    torch.save(checkpoint, path)
                    reg["refs"]["initial_checkpoint"] = validation.file_ref(path)
                validation.write_json(registration, reg)
                with mock.patch("mario_rl.model.ActorCritic", TinyActor), self.assertRaises(ValueError):
                    validation.audit_registration(registration, environment_kind="synthetic_fixture")

    def test_real_mode_rejects_injected_model_or_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            context = fake_context(directory)
            context["environment_kind"] = "real_nes"
            with self.assertRaisesRegex(ValueError, "Injected"):
                validation.Validator(context, model=Policy())
            self.assertFalse(context["output"].exists())

    def test_real_video_encoder_is_counted_without_any_game(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.mp4"
            writer = validation.video_writer(path)
            for value in range(3):
                writer.append_data(np.full((240, 256, 3), value * 100, dtype=np.uint8))
            writer.close()
            metadata = validation.inspect_video(path, 3, (256, 240))
            self.assertEqual(metadata["frames"], 3)
            self.assertAlmostEqual(metadata["duration_seconds"], .2)
            with self.assertRaisesRegex(ValueError, "frames/fps/resolution"):
                validation.inspect_video(path, 4, (256, 240))

    def test_invalid_source_reference_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "value.json"
            p.write_text("{}")
            link = Path(directory) / "link.json"
            link.symlink_to(p)
            with self.assertRaisesRegex(ValueError, "path and sha256"):
                validation.checked({"path": str(p), "bytes": 2})
            with self.assertRaisesRegex(ValueError, "Symlink"):
                validation.checked({"path": str(link), "sha256": validation.digest(b"{}")})


if __name__ == "__main__":
    unittest.main()
