"""Small, deterministic checks for multi-level evaluation and its evidence files."""
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mario_rl.evaluate import evaluate, telemetry


class FakeEvaluationEnv:
    """An RGB-producing environment with NES-like NumPy telemetry scalars."""

    def __init__(self, *, terminated=True, truncated=False, flag=False,
                 time_left=299, reason=None, x=100):
        self.terminated = terminated
        self.truncated = truncated
        self.flag = flag
        self.time_left = time_left
        self.reason = reason
        self.x = x
        self.closed = False
        self.reset_seeds = []
        self.calls = 0

    def _info(self, *, initial=False):
        info = dict(world=np.uint8(2), stage=np.uint8(2),
                    x_pos=np.int32(40 if initial else self.x),
                    y_pos=np.uint8(80), life=np.uint8(2),
                    time=np.int32(400 if initial else self.time_left),
                    score=np.int32(0 if initial else 100), coins=np.uint8(0),
                    flag_get=np.bool_(False if initial else self.flag), status="small")
        if not initial:
            info["raw_reward"] = np.float32(2.5)
            info["frames_advanced"] = np.int64(4)
            if self.reason:
                info["termination_reason"] = self.reason
        return info

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.calls = 0
        return np.zeros((4, 84, 84), np.float32), self._info(initial=True)

    def step(self, action):
        self.calls += 1
        return (np.zeros((4, 84, 84), np.float32), 1.25,
                self.terminated, self.truncated, self._info())

    def render(self):
        return np.zeros((240, 256, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class EvaluationTests(unittest.TestCase):
    def run_evaluation(self, folder, env, *, checkpoint=None, episodes=1, max_steps=4):
        args = SimpleNamespace(policy="local" if checkpoint else "random",
                               checkpoint=checkpoint, device="cpu", world=2, stage=2,
                               seed=7, episodes=episodes, max_steps=max_steps,
                               record=False, output=str(folder))
        model = object() if checkpoint else None
        with patch("mario_rl.evaluate.make_env", return_value=env) as factory, \
             patch("mario_rl.evaluate.policy_model", return_value=model), \
             patch("mario_rl.evaluate.choose_action", return_value=0), \
             contextlib.redirect_stdout(io.StringIO()):
            result = evaluate(args)
        factory.assert_called_once_with(world=2, stage=2, seed=7, reward_mode="full", castle_path_rules=False)
        self.assertTrue(env.closed)
        return result

    def test_non_first_level_checkpoint_identity_and_json_numpy_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "specialist.pt"
            payload = b"A mocked checkpoint: policy loading is patched in this test."
            checkpoint.write_bytes(payload)
            env = FakeEvaluationEnv(flag=True)
            result = self.run_evaluation(root / "evaluation", env,
                                         checkpoint=str(checkpoint), episodes=3)
            self.assertEqual(result["environment"], "SuperMarioBros-2-2-v0")
            self.assertEqual((result["world"], result["stage"]), (2, 2))
            self.assertEqual(result["max_steps"], 4)
            self.assertEqual(result["policy"], "checkpoint")
            self.assertEqual(result["checkpoint"], str(checkpoint.resolve()))
            self.assertEqual(result["checkpoint_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(env.reset_seeds, [7, 8, 9])
            saved = json.loads((root / "evaluation" / "results.json").read_text())
            self.assertEqual(saved["summary"]["wins"], 3)
            self.assertEqual(saved["episodes"][0]["initial_info"]["life"], 2)
            self.assertIsInstance(saved["episodes"][0]["initial_info"]["life"], int)
            self.assertIs(saved["episodes"][0]["final_info"]["flag_get"], True)
            self.assertEqual(saved["episodes"][0]["raw_return"], 2.5)
            with (root / "evaluation" / "episodes.csv").open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(json.loads(row["final_info"])["world"], 2)
            self.assertTrue((root / "evaluation" / "final-frame.png").is_file())

    def test_end_reasons_distinguish_clear_death_game_clock_and_limits(self):
        cases = [
            ({"flag": True, "time_left": 0}, "level_clear", False),
            ({"time_left": 0}, "game_timer_expired", False),
            ({"time_left": 250}, "environment_terminal_without_clear", False),
            ({"terminated": False, "truncated": True}, "environment_time_limit", False),
            ({"terminated": False}, "evaluation_step_limit", True),
            ({"reason": "wrong_castle_path"}, "wrong_castle_path", False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, (kwargs, expected, evaluation_limit) in enumerate(cases):
                with self.subTest(expected=expected):
                    env = FakeEvaluationEnv(**kwargs)
                    result = self.run_evaluation(Path(directory) / str(index), env)
                    row = result["episodes"][0]
                    self.assertEqual(row["termination_reason"], expected)
                    self.assertEqual(row["evaluation_limit"], evaluation_limit)
                    self.assertEqual(row["steps"], 4 if evaluation_limit else 1)
                    if evaluation_limit:
                        self.assertEqual(row["steps_since_best_x"], 3)

    def test_random_strategy_has_no_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_evaluation(Path(directory), FakeEvaluationEnv())
            self.assertEqual(result["policy"], "random")
            self.assertIsNone(result["checkpoint"])
            self.assertIsNone(result["checkpoint_sha256"])
            self.assertFalse(result["deterministic_policy"])

    def test_cli_passes_requested_world_and_stage_to_evaluation(self):
        from mario_rl.__main__ import main
        with patch("sys.argv", ["mario_rl", "evaluate", "--world", "2", "--stage", "2",
                                "--episodes", "3", "--max-steps", "3000"]), \
             patch("mario_rl.evaluate.evaluate") as mocked_evaluate:
            main()
        args = mocked_evaluate.call_args.args[0]
        self.assertEqual((args.world, args.stage, args.episodes, args.max_steps), (2, 2, 3, 3000))

    def test_telemetry_preserves_scalar_values_as_json_types(self):
        values = telemetry({"life": np.uint8(2), "flag_get": np.bool_(False),
                            "x_pos": np.int64(1234), "reward": np.float32(0.5),
                            "status": "small"})
        self.assertEqual(json.loads(json.dumps(values)),
                         {"life": 2, "flag_get": False, "x_pos": 1234,
                          "reward": 0.5, "status": "small"})


if __name__ == "__main__":
    unittest.main()
