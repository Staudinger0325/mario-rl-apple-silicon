"""Regression tests for checkpoint preprocessing and episode boundaries."""
from pathlib import Path
import pickle
import tempfile
import unittest

import gym
import numpy as np
import torch

from mario_rl.env import MarioEnv, make_env
from mario_rl.model import ActorCritic, load_model, select_device
from mario_rl.weights import ensure_weights


class FakeNES:
    """A deterministic emulator substitute whose individual frames are visible."""

    def __init__(self, terminal_at=None, legacy_time_limit=False, *,
                 clear=False, initial_score=0, fixed_x=None, y_pos=80):
        self.action_space = gym.spaces.Discrete(7)
        self.terminal_at = terminal_at
        self.legacy_time_limit = legacy_time_limit
        self.clear = clear
        self.initial_score = initial_score
        self.fixed_x = fixed_x
        self.y_pos = y_pos
        self._area = np.uint8(3)
        self.reset_count = 0
        self.index = 0
        self.unwrapped = self

    def _get_info(self):
        return {"score": self.initial_score + self.index * 10,
                "x_pos": self.fixed_x if self.fixed_x is not None else 40 + self.index,
                "y_pos": self.y_pos,
                "flag_get": self.clear and self.index == self.terminal_at,
                "time": 400}

    def reset(self, **kwargs):
        self.reset_count += 1
        self.index = 0
        return np.zeros((240, 256, 3), dtype=np.uint8), self._get_info()

    def step(self, action):
        self.index += 1
        intensity = [20, 50, 90, 40, 60, 70, 140, 100][(self.index - 1) % 8]
        frame = np.full((240, 256, 3), intensity, dtype=np.uint8)
        done = self.index == self.terminal_at
        info = self._get_info()
        if self.legacy_time_limit:
            info["TimeLimit.truncated"] = done
        return frame, 1.0, done, info

    def close(self):
        pass


class UntrustedPayload:
    pass


class EnvironmentTests(unittest.TestCase):
    def test_preprocessing_is_rolling_decision_stack_not_four_raw_frames(self):
        env = MarioEnv(_base_env=FakeNES())
        obs, _ = env.reset()
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(obs.shape, (4, 84, 84))
        obs, reward, terminated, truncated, _ = env.step(1)
        self.assertFalse(terminated or truncated)
        np.testing.assert_array_equal(obs[:3], 0)
        np.testing.assert_allclose(obs[3], 90 / 255)
        self.assertAlmostEqual(reward, 0.5)
        obs2, *_ = env.step(1)
        np.testing.assert_allclose(obs2[2], 90 / 255)
        np.testing.assert_allclose(obs2[3], 140 / 255)
        np.testing.assert_allclose(obs[3], 90 / 255)  # Old obs is not aliased.
        self.assertEqual(env.render().shape, (240, 256, 3))
        env.close()

    def test_terminal_observation_is_not_an_autoreset_observation(self):
        base = FakeNES(terminal_at=2)
        env = MarioEnv(_base_env=base)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(2)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(base.reset_count, 1)
        self.assertEqual(info["frames_advanced"], 2)
        self.assertAlmostEqual(reward, -4.75)
        np.testing.assert_allclose(obs[-1], 50 / 255)
        with self.assertRaises(RuntimeError):
            env.step(2)
        env.close()

    def test_wrapper_timeout_is_truncation_with_bootstrappable_observation(self):
        base = FakeNES()
        env = MarioEnv(_base_env=base, max_steps=1)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["TimeLimit.truncated"])
        self.assertEqual(base.reset_count, 1)
        self.assertAlmostEqual(reward, 0.5)  # No death penalty on timeout.
        np.testing.assert_allclose(obs[-1], 90 / 255)
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.close()

    def test_legacy_gym_time_limit_is_not_a_terminal_failure(self):
        env = MarioEnv(_base_env=FakeNES(terminal_at=2, legacy_time_limit=True))
        env.reset()
        _, reward, terminated, truncated, _ = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(reward, 0.25)
        env.close()

    def test_action_validation(self):
        env = MarioEnv(_base_env=FakeNES())
        env.reset()
        for action in (-1, 7, 1.5, "right"):
            with self.assertRaises(ValueError):
                env.step(action)
        env.step(np.int64(1))
        env.close()

    def test_real_nes_reset_step_and_seeded_action_space(self):
        env = make_env(seed=42, max_steps=1)
        try:
            obs, info = env.reset()
            self.assertEqual(env.action_space.n, 7)
            self.assertEqual(info["x_pos"], 40)
            self.assertTrue(env.observation_space.contains(obs))
            action = env.action_space.sample()
            env.reset(seed=42)
            self.assertEqual(action, env.action_space.sample())
            obs, _, terminated, truncated, info = env.step(1)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(info["frames_advanced"], 4)
            self.assertIsInstance(info["game_area"], int)
            self.assertEqual(info["reward_mode"], "full")
            self.assertTrue(info["castle_path_rules"])
            self.assertEqual(env.render().dtype, np.uint8)
        finally:
            env.close()

    def test_reward_modes_preserve_counterfactual_components(self):
        cases = [(None, False, 0.4, 0.1, 0.0),
                 (2, False, 0.2, 0.05, -5.0),
                 (2, True, 0.2, 0.05, 5.0)]
        for terminal_at, clear, native, score, terminal in cases:
            for mode in ("native", "progress_terminal", "progress_score", "full"):
                with self.subTest(mode=mode, terminal_at=terminal_at, clear=clear):
                    env = MarioEnv(_base_env=FakeNES(terminal_at=terminal_at, clear=clear),
                                   reward_mode=mode, castle_path_rules=False)
                    env.reset()
                    _, reward, ended, truncated, info = env.step(1)
                    expected = native + (terminal if mode in ("progress_terminal", "full") else 0)
                    expected += score if mode in ("progress_score", "full") else 0
                    self.assertAlmostEqual(reward, expected)
                    self.assertAlmostEqual(info["raw_reward"], native * 10)
                    self.assertAlmostEqual(info["reward_native"], native)
                    self.assertAlmostEqual(info["reward_score"], score)
                    self.assertAlmostEqual(info["reward_terminal"], terminal)
                    self.assertEqual(info["reward_castle_rule"], 0)
                    self.assertAlmostEqual(info["shaped_reward"], reward)
                    self.assertAlmostEqual(sum(info["applied_reward_components"].values()), reward)
                    self.assertEqual(info["reward_components"],
                                     {key: info[f"reward_{key}"]
                                      for key in ("native", "score", "terminal", "castle_rule")})
                    self.assertEqual(ended, terminal_at is not None)
                    self.assertFalse(truncated)
                    self.assertEqual(info["reward_mode"], mode)
                    self.assertFalse(info["castle_path_rules"])
                    env.close()

    def test_reward_components_accumulate_within_decision_and_reset_between_decisions(self):
        env = MarioEnv(_base_env=FakeNES(initial_score=100), castle_path_rules=False)
        _, initial = env.reset()
        self.assertEqual(initial["game_area"], 3)
        self.assertIsInstance(initial["game_area"], int)
        for _ in range(2):
            _, reward, _, _, info = env.step(1)
            self.assertAlmostEqual(info["raw_reward"], 4)
            self.assertAlmostEqual(info["reward_native"], 0.4)
            self.assertAlmostEqual(info["reward_score"], 0.1)
            self.assertAlmostEqual(reward, 0.5)
        env.reset()
        _, reward, _, _, info = env.step(1)
        self.assertAlmostEqual(info["reward_score"], 0.1)
        self.assertAlmostEqual(reward, 0.5)
        env.close()

    def test_no_reward_mode_treats_time_limit_as_death(self):
        for mode in ("native", "progress_terminal", "progress_score", "full"):
            for legacy in (False, True):
                with self.subTest(mode=mode, legacy=legacy):
                    base = FakeNES(terminal_at=2 if legacy else None,
                                   legacy_time_limit=legacy)
                    env = MarioEnv(_base_env=base, max_steps=1, reward_mode=mode,
                                   castle_path_rules=False)
                    env.reset()
                    _, reward, terminated, truncated, info = env.step(1)
                    count = 2 if legacy else 4
                    self.assertFalse(terminated)
                    self.assertTrue(truncated)
                    self.assertEqual(info["reward_terminal"], 0.0)
                    expected = count / 10 + (count / 40 if mode in ("progress_score", "full") else 0)
                    self.assertAlmostEqual(reward, expected)
                    env.close()

    def test_castle_rule_can_be_disabled_without_faking_clear(self):
        for mode in ("native", "progress_terminal", "progress_score", "full"):
            env = MarioEnv(world=4, stage=4, _base_env=FakeNES(),
                           reward_mode=mode, castle_path_rules=False)
            env.reset()
            _, reward, terminated, truncated, info = env.step(1)
            self.assertFalse(terminated or truncated)
            self.assertFalse(info["flag_get"])
            self.assertFalse(info["castle_path_rule_triggered"])
            self.assertEqual(info["reward_castle_rule"], 0.0)
            self.assertEqual(info["frames_advanced"], 4)
            self.assertNotIn("termination_reason", info)
            self.assertAlmostEqual(reward, 0.5 if mode in ("progress_score", "full") else 0.4)
            env.close()

    def test_legacy_full_castle_adjustment_is_separate_from_game_terminal_reward(self):
        for world, x, y, expected, adjustment in [(4, 41, 80, -5.0, -5.125),
                                                  (7, 600, 160, -4.875, -5.0)]:
            with self.subTest(world=world):
                env = MarioEnv(world=world, stage=4,
                               _base_env=FakeNES(fixed_x=x, y_pos=y))
                env.reset()
                _, reward, terminated, truncated, info = env.step(1)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertFalse(info["flag_get"])
                self.assertTrue(info["castle_path_rule_triggered"])
                self.assertEqual(info["termination_reason"], "wrong_castle_path")
                self.assertEqual(info["reward_terminal"], 0.0)
                self.assertAlmostEqual(info["reward_castle_rule"], adjustment)
                self.assertAlmostEqual(reward, expected)
                env.close()
        for mode in ("native", "progress_terminal", "progress_score"):
            env = MarioEnv(world=4, stage=4, reward_mode=mode, _base_env=FakeNES())
            env.reset()
            _, reward, terminated, _, info = env.step(1)
            self.assertTrue(terminated)
            self.assertAlmostEqual(reward, 0.125 if mode == "progress_score" else 0.1)
            self.assertAlmostEqual(info["reward_castle_rule"], -5.125)
            self.assertEqual(info["applied_reward_components"]["castle_rule"], 0)
            env.close()

    def test_invalid_environment_parameters_fail_before_initialization(self):
        invalid = [{"world": value} for value in (0, 9, 1.5, True, "1")]
        invalid += [{"stage": value} for value in (0, 5, 1.5, False)]
        invalid += [{"max_steps": value} for value in (0, -1, 1.5, True)]
        invalid += [{"reward_mode": value} for value in ("", "unknown", ["full"])]
        invalid += [{"castle_path_rules": value} for value in ("false", 0, 1)]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MarioEnv(_base_env=FakeNES(), **kwargs)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_bundled_local_state_dict_and_training_checkpoint_match(self):
        model = load_model(ensure_weights(), device=torch.device("cpu"))
        self.assertFalse(model.training)
        x = torch.zeros(2, 4, 84, 84)
        with torch.inference_mode():
            logits, values = model(x)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(tuple(values.shape), (2, 1))
        self.assertTrue(torch.isfinite(logits).all())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            torch.save({"model": model.state_dict(), "global_step": 123}, path)
            resumed = load_model(path)
            with torch.inference_mode():
                self.assertTrue(torch.equal(logits, resumed(x)[0]))

    def test_loader_rejects_arbitrary_pickle_globals(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "untrusted.pt"
            torch.save(UntrustedPayload(), path)
            with self.assertRaises(pickle.UnpicklingError):
                load_model(path)

    def test_cpu_device_is_always_available(self):
        self.assertEqual(select_device("cpu"), torch.device("cpu"))
        self.assertIn(select_device().type, {"cpu", "mps", "cuda"})


if __name__ == "__main__":
    unittest.main()
