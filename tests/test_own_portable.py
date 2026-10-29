"""Portable CLI and actual synthetic search -> BC -> policy-audit contracts."""
import builtins
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import explore_own_prefixes as search
from scripts import train_own_trajectory_bc as bc
from scripts import validate_own_trajectory_policy as validation
from tests.test_own_prefixes import SyntheticEnv, config
from tests.test_own_trajectory_bc import TinyActor
from tests.test_own_trajectory_validation import VideoEnv, Writer, inspect


def portable_search(folder, *, seed=73, env_seed=9, length=5):
    engine = search.Search(config(Path(folder) / "search", seed=seed, env_seed=env_seed),
                           lambda: SyntheticEnv(length=length), environment_kind="synthetic_fixture")
    with mock.patch.object(engine.rng, "randrange", return_value=0):
        result = engine.run()
    assert result["trajectory_verified"], result
    return engine.output


class PortableOwnTrajectoryTests(unittest.TestCase):
    def test_new_seed_and_length_pass_real_producer_consumer_chain(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary).resolve()
            source = portable_search(root)
            request, output = root / "bc-request.json", root / "bc"
            bc.prepare_from_search(source, request, output, device="cpu",
                                   environment_kind="synthetic_fixture")
            reg = json.loads(request.read_text())
            self.assertEqual(reg["recipe"]["seed"], 73)
            self.assertEqual(reg["recipe"]["env_seed"], 9)
            self.assertEqual(reg["trajectory_steps"], 5)
            self.assertFalse(reg["historical_reproduction_claim"])
            self.assertFalse(output.exists())
            result = bc.run_bc(request, environment_kind="synthetic_fixture",
                               env_factory=lambda: SyntheticEnv(length=5), model_factory=TinyActor)
            self.assertEqual(result["status"], "dataset_fit_criterion_met")
            self.assertEqual(result["model_config"]["seed"], 73)
            self.assertEqual(result["model_config"]["env_seed"], 9)
            self.assertEqual(result["matched_transitions"], 5)
            self.assertEqual(result["bc_optimizer_steps"], 5)
            self.assertEqual(result["ppo_decisions"], 0)
            evaluation = root / "validation-request.json"
            with mock.patch("mario_rl.model.ActorCritic", TinyActor):
                validation.prepare_from_bc(output, evaluation, root / "validation",
                                            environment_kind="synthetic_fixture")
                context = validation.audit_registration(evaluation, environment_kind="synthetic_fixture")
            self.assertFalse(context["output"].exists())
            actor = TinyActor()
            actor.load_state_dict(context["checkpoint"]["model"])
            result = validation.Validator(context, model=actor,
                env_factory=lambda seed: VideoEnv(length=5), writer_factory=Writer,
                video_inspector=inspect).run()
            self.assertEqual(result["status"], "synthetic_fixture_passed")
            self.assertEqual(result["clear_episodes"], 5)
            self.assertEqual(result["issued_env_step_calls"], 25)
            self.assertFalse(result["network_policy_validated"])
            self.assertEqual(context["initial"]["lineage"]["initialization"]["seed"], 73)

    def test_portable_cli_preexisting_stop_never_imports_nes_or_old_review(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary).resolve()
            stop = root / "STOP"
            stop.touch()
            original_import = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.startswith(("mario_rl.env", "gym_super_mario_bros", "nes_py")):
                    raise AssertionError("STOP must prevent real environment import")
                return original_import(name, *args, **kwargs)
            with mock.patch("builtins.__import__", side_effect=guarded):
                code = search.main(["--portable", "--output", str(root / "search"),
                    "--seed", "73", "--budget", "10", "--max-hours", "0.01", "--stop-file", str(stop)])
            self.assertEqual(code, 2)
            manifest = json.loads((root / "search/manifest.json").read_text())
            self.assertEqual(manifest["counts"]["total_decisions"], 0)
            self.assertFalse(manifest["trajectory_verified"])
            self.assertIsNotNone(manifest["config"]["deadline"])
            review = json.loads((root / "search/launch-review.json").read_text())
            self.assertEqual(review["decision"], "portable_new_experiment")
            self.assertNotIn("entropy01_complete_failure_reviewed", review)
            with self.assertRaises((ValueError, FileExistsError)):
                search.main(["--portable", "--output", str(root / "search")])

    def test_prepare_refuses_partial_tampered_and_existing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = portable_search(root)
            request, output = root / "request.json", root / "bc"
            bc.prepare_from_search(source, request, output, device="cpu", environment_kind="synthetic_fixture")
            original = request.read_bytes()
            with self.assertRaisesRegex(ValueError, "new path"):
                bc.prepare_from_search(source, request, output, device="cpu", environment_kind="synthetic_fixture")
            self.assertEqual(request.read_bytes(), original)
            candidate = source / "candidate-trajectory.json"
            candidate.write_bytes(candidate.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "does not bind"):
                bc.prepare_from_search(source, root / "bad.json", output, device="cpu",
                                       environment_kind="synthetic_fixture")
            self.assertFalse((root / "bad.json").exists())
            self.assertFalse(output.exists())

    def test_portable_still_rejects_environment_kind_and_seed_identity_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = portable_search(root)
            request = root / "request.json"
            bc.prepare_from_search(source, request, root / "bc", device="cpu",
                                   environment_kind="synthetic_fixture")
            with self.assertRaisesRegex(ValueError, "verified"):
                bc.validate_registration(request)  # Synthetic data cannot become real evidence.
            reg = json.loads(request.read_text())
            reg["recipe"]["seed"] = 74
            request.write_text(json.dumps(reg))
            with self.assertRaisesRegex(ValueError, "verified"):
                bc.validate_registration(request, environment_kind="synthetic_fixture")

    def test_deadline_is_new_bounded_and_expired_request_is_rejected(self):
        before = datetime.now(timezone.utc)
        deadline = datetime.fromisoformat(search.portable_deadline(.01))
        self.assertGreater(deadline, before)
        self.assertLessEqual((deadline - before).total_seconds(), 37)
        for hours, value in [(0, None), (9, None), (True, None), (.1, "2000-01-01T00:00:00+00:00"),
                             (.1, "2099-01-01T00:00:00")]:
            with self.subTest(hours=hours, deadline=value), self.assertRaises(ValueError):
                search.portable_deadline(hours, value)

    def test_run_cli_rejects_silently_ignored_budget_overrides(self):
        for entry in (bc.main, validation.main):
            with self.subTest(entry=entry.__module__), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    entry(["--registration", "does-not-exist.json", "--max-hours", "0.1"])
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
