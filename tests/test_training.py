"""Numerical correctness checks for PPO boundaries and resumable checkpoints."""

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mario_rl.train import (
    capture_rng_state, clipped_policy_loss, critic_value_loss, compute_gae, create_optimizer,
    restore_rng_state, save_checkpoint, build_parser, inherit_protocol, evaluation_score,
    update_evaluation_state, evaluate_training_policy, prune_checkpoints, model_digest,
    _open_csv,
)


class AdvantageTests(unittest.TestCase):
    def test_terminal_does_not_bootstrap_or_leak_new_episode(self):
        advantages, returns = compute_gae(
            [[1], [100]], [[2], [0]], [[999], [0]],
            [[True], [True]], [[False], [False]], gamma=0.9, gae_lambda=1.0)
        np.testing.assert_allclose(advantages[:, 0], [-1, 100])
        np.testing.assert_allclose(returns[:, 0], [1, 100])

    def test_truncation_bootstraps_final_observation_without_crossing_reset(self):
        advantages, returns = compute_gae(
            [[1], [100]], [[2], [0]], [[10], [0]],
            [[False], [True]], [[True], [False]], gamma=0.9, gae_lambda=1.0)
        np.testing.assert_allclose(advantages[:, 0], [8, 100])
        np.testing.assert_allclose(returns[:, 0], [10, 100])

    def test_unfinished_rollout_bootstraps_and_propagates(self):
        advantages, _ = compute_gae(
            [[1], [2]], [[3], [4]], [[4], [5]],
            [[False], [False]], [[False], [False]], gamma=0.9, gae_lambda=0.8)
        np.testing.assert_allclose(advantages[:, 0], [3.4, 2.5], rtol=1e-6)

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            compute_gae([1], [[0]], [[0]], [[False]], [[False]])


class PolicyLossTests(unittest.TestCase):
    def test_clipping_signs_are_pessimistic(self):
        # Positive advantage clips large ratios, negative clips small ratios.
        old = torch.zeros(2)
        new = torch.tensor([1.5, 0.5]).log()
        loss = clipped_policy_loss(new, old, torch.tensor([2.0, -2.0]), 0.2)
        self.assertAlmostEqual(loss.item(), -0.4, places=6)

    def test_unclipped_gradient_is_nonzero(self):
        new = torch.tensor([0.0], requires_grad=True)
        loss = clipped_policy_loss(new, torch.zeros(1), torch.ones(1))
        loss.backward()
        self.assertAlmostEqual(new.grad.item(), -1.0)


class ValueLossTests(unittest.TestCase):
    def test_default_mse_matches_original_value_and_gradient_exactly(self):
        values = torch.tensor([-7.0, -0.5, 0.25, 12.0], requires_grad=True)
        returns = torch.tensor([1.0, 0.0, 0.5, -2.0])
        reference = 0.5 * (values - returns).square().mean()
        expected_gradient, = torch.autograd.grad(reference, values)
        rng = torch.get_rng_state().clone()
        actual = critic_value_loss(values, returns)
        actual_gradient, = torch.autograd.grad(actual, values)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_huber_matches_half_mse_within_unit_residual(self):
        values = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], requires_grad=True)
        returns = torch.zeros_like(values)
        mse = critic_value_loss(values, returns, "mse")
        huber = critic_value_loss(values, returns, "huber")
        mse_gradient, = torch.autograd.grad(mse, values)
        huber_gradient, = torch.autograd.grad(huber, values)
        torch.testing.assert_close(huber, mse, rtol=0, atol=0)
        torch.testing.assert_close(huber_gradient, mse_gradient, rtol=0, atol=0)

    def test_huber_linear_tails_bound_each_residual_gradient(self):
        values = torch.tensor([-100.0, -2.0, 0.5, 3.0, 200.0], requires_grad=True)
        loss = critic_value_loss(values, torch.zeros_like(values), "huber")
        # Huber beta=1: half the square below one; abs(error)-0.5 above.
        self.assertAlmostEqual(loss.item(), (99.5 + 1.5 + 0.125 + 2.5 + 199.5) / 5,
                               places=5)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([-0.2, -0.2, 0.1, 0.2, 0.2]))
        self.assertLessEqual(float(values.grad.abs().max()), 1 / len(values) + 1e-7)

    def test_unknown_value_loss_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown value loss"):
            critic_value_loss(torch.zeros(1), torch.zeros(1), "typo")


class CheckpointTests(unittest.TestCase):
    def test_only_numbered_archives_are_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["initial.pt", "best.pt", "last.pt", "checkpoint_notes.pt",
                     "checkpoint_8.pt", "checkpoint_16.pt", "checkpoint_24.pt"]
            for name in names:
                (root / name).write_bytes(b"checkpoint")
            prune_checkpoints(root, 1)
            self.assertFalse((root / "checkpoint_8.pt").exists())
            self.assertFalse((root / "checkpoint_16.pt").exists())
            for name in names[:4] + ["checkpoint_24.pt"]:
                self.assertTrue((root / name).exists())

    def test_model_hash_tracks_tensor_values_not_serialization(self):
        model = torch.nn.Linear(2, 1)
        before = model_digest(model)
        clone = torch.nn.Linear(2, 1)
        clone.load_state_dict(model.state_dict())
        self.assertEqual(before, model_digest(clone))
        with torch.no_grad():
            clone.bias.add_(0.1)
        self.assertNotEqual(before, model_digest(clone))

    def test_resume_preserves_learning_rate_unless_explicitly_overridden(self):
        model = torch.nn.Linear(2, 1)
        state = torch.optim.Adam(model.parameters(), lr=0.0007).state_dict()
        resumed = create_optimizer(model, optimizer_state=state)
        self.assertEqual(resumed.param_groups[0]["lr"], 0.0007)
        overridden = create_optimizer(model, learning_rate=0.0001, optimizer_state=state)
        self.assertEqual(overridden.param_groups[0]["lr"], 0.0001)

    def test_checkpoint_is_safe_to_load_and_preserves_optimizer_and_rng(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, model, optimizer, 128, 2, {"world": 1, "stage": 1})
            expected = (random.random(), float(np.random.random()), torch.rand(3))
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["global_step"], 128)
            self.assertEqual(checkpoint["update"], 2)
            self.assertTrue(checkpoint["optimizer"]["state"])
            self.assertFalse(path.with_suffix(".pt.tmp").exists())
            restore_rng_state(checkpoint["rng"])
            self.assertEqual(random.random(), expected[0])
            self.assertEqual(float(np.random.random()), expected[1])
            torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)

    def test_rng_capture_round_trip(self):
        state = capture_rng_state()
        expected = np.random.randint(100, size=5)
        restore_rng_state(state)
        np.testing.assert_array_equal(np.random.randint(100, size=5), expected)


class TrainingProtocolTests(unittest.TestCase):
    def test_default_initialization_is_scratch(self):
        self.assertEqual(build_parser().parse_args([]).init, "scratch")

    def test_new_run_records_resolved_mse_or_explicit_huber(self):
        for supplied, expected in [([], "mse"), (["--value-loss", "huber"], "huber")]:
            args = build_parser().parse_args(supplied)
            inherit_protocol(args)
            self.assertEqual(vars(args)["value_loss"], expected)

    def test_resume_inherits_value_loss_and_legacy_defaults_to_mse(self):
        for config, expected in [({}, "mse"), ({"value_loss": "mse"}, "mse"),
                                 ({"value_loss": "huber"}, "huber")]:
            args = build_parser().parse_args(["--resume", "last.pt"])
            inherit_protocol(args, config)
            self.assertEqual(args.value_loss, expected)

    def test_resume_accepts_only_explicit_matching_value_loss(self):
        for existing in ("mse", "huber"):
            matching = build_parser().parse_args(["--resume", "last.pt", "--value-loss", existing])
            inherit_protocol(matching, {"value_loss": existing})
            other = "huber" if existing == "mse" else "mse"
            mismatched = build_parser().parse_args(["--resume", "last.pt", "--value-loss", other])
            with self.assertRaisesRegex(ValueError, "Resume value_loss must match"):
                inherit_protocol(mismatched, {"value_loss": existing})
        legacy = build_parser().parse_args(["--resume", "last.pt", "--value-loss", "huber"])
        with self.assertRaisesRegex(ValueError, "Resume value_loss must match"):
            inherit_protocol(legacy, {})

    def test_resume_rejects_corrupt_saved_value_loss(self):
        args = build_parser().parse_args(["--resume", "last.pt"])
        with self.assertRaisesRegex(ValueError, "Unknown value loss"):
            inherit_protocol(args, {"value_loss": "typo"})

    def test_new_run_defaults_and_resume_inheritance(self):
        args = SimpleNamespace(resume=None, reward_mode=None, castle_path_rules=None)
        inherit_protocol(args)
        self.assertEqual(args.reward_mode, "full")
        self.assertFalse(args.castle_path_rules)
        args = SimpleNamespace(resume="last.pt", reward_mode=None, castle_path_rules=None)
        inherit_protocol(args, {"reward_mode": "native", "castle_path_rules": False})
        self.assertEqual(args.reward_mode, "native")
        self.assertFalse(args.castle_path_rules)

    def test_resume_rejects_reward_mode_switch(self):
        args = SimpleNamespace(resume="last.pt", reward_mode="full", castle_path_rules=None)
        with self.assertRaisesRegex(ValueError, "reward_mode"):
            inherit_protocol(args, {"reward_mode": "native", "castle_path_rules": False})

    def test_legacy_resume_retains_castle_rules(self):
        args = SimpleNamespace(resume="legacy.pt", reward_mode=None, castle_path_rules=None)
        inherit_protocol(args, {})
        self.assertEqual(args.reward_mode, "full")
        self.assertTrue(args.castle_path_rules)

    def test_csv_schema_upgrade_preserves_old_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.csv"
            path.write_text("return,length\n3,10\n", encoding="utf-8")
            stream, writer = _open_csv(path, ["return", "length", "raw_return"])
            writer.writerow({"return": 4, "length": 11, "raw_return": 40})
            stream.close()
            self.assertEqual(path.read_text().splitlines(),
                             ["return,length,raw_return", "3,10,", "4,11,40"])


class PeriodicEvaluationTests(unittest.TestCase):
    @staticmethod
    def rows(wins, reward=100.0):
        return [{"flag_get": win, "raw_return": reward, "steps": 10, "complete": True}
                for win in wins]

    def test_success_rate_has_priority_over_raw_return(self):
        state = {}
        self.assertTrue(update_evaluation_state(state, self.rows([False], 9999), "p", 10, "a"))
        self.assertTrue(update_evaluation_state(state, self.rows([True], -1), "p", 20, "b"))
        self.assertEqual(state["best_evaluation"]["model_sha256"], "b")
        self.assertFalse(update_evaluation_state(state, self.rows([False], 99999), "p", 30, "c"))

    def test_protocol_change_resets_comparison_and_consecutive_count(self):
        state = {}
        update_evaluation_state(state, self.rows([True]), "old", 10, "a")
        update_evaluation_state(state, self.rows([True]), "old", 20, "b")
        self.assertEqual(state["consecutive_clear_evals"], 2)
        update_evaluation_state(state, self.rows([True]), "new", 30, "c")
        self.assertEqual(state["consecutive_clear_evals"], 1)
        self.assertEqual(state["best_evaluation"]["protocol_sha256"], "new")

    def test_incomplete_evaluation_does_not_create_success_history(self):
        state = {}
        rows = [{"flag_get": True, "raw_return": 100, "steps": 1, "complete": False}]
        self.assertIsNone(evaluation_score(rows))
        self.assertFalse(update_evaluation_state(state, rows, "p", 0, "m"))
        self.assertEqual(state, {})

    def test_evaluation_preserves_rng_and_honors_stop(self):
        class TinyModel(torch.nn.Module):
            def forward(self, obs):
                return torch.zeros((len(obs), 2)), torch.zeros((len(obs), 1))

        class TinyEnvironment:
            def reset(self, seed=None):
                random.random()
                np.random.random()
                torch.rand(1)
                return np.zeros((4, 2, 2), dtype=np.float32), {"x_pos": 0}

            def step(self, action):
                return np.zeros((4, 2, 2), dtype=np.float32), 1.0, True, False, {
                    "flag_get": True, "x_pos": 1, "raw_reward": 10,
                    "reward_components": {"native": 1}, "applied_reward_components": {"native": 1}}

        model = TinyModel()
        original = capture_rng_state()
        expected = (random.random(), np.random.random(), torch.rand(1))
        restore_rng_state(original)
        rows, stopped = evaluate_training_policy(model, TinyEnvironment(), "cpu", 2, 4, 0, lambda: None)
        self.assertIsNone(stopped)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["flag_get"] for row in rows))
        self.assertEqual(rows[0]["raw_return"], 10)
        self.assertTrue(model.training)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        torch.testing.assert_close(torch.rand(1), expected[2], rtol=0, atol=0)
        rows, stopped = evaluate_training_policy(model, TinyEnvironment(), "cpu", 2, 4, 0,
                                                 lambda: "max_hours")
        self.assertEqual(rows, [])
        self.assertEqual(stopped, "max_hours")


if __name__ == "__main__":
    unittest.main()
