"""Only synthetic deterministic environments; never construct NES or a model."""
import csv
import builtins
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import explore_own_prefixes as search


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class SyntheticEnv:
    def __init__(self, length=3, clear=True, truncated=False, on_step=None, on_reset=None):
        self.action_space = SimpleNamespace(n=7)
        self.actions = search.ACTIONS
        self.length, self.is_clear, self.is_truncated = length, clear, truncated
        self.on_step, self.on_reset = on_step, on_reset
        self.calls = self.reset_calls = 0
        self.seen_actions = []
        self._closed = False

    def info(self):
        return {"world": 1, "stage": 3, "game_area": 4, "x_pos": self._current_x,
                "y_pos": 80, "time": 300 - self._steps, "score": self._score,
                "episode_step": self._steps, "reward_mode": "full",
                "castle_path_rules": False, "castle_path_rule_triggered": False,
                "flag_get": self.is_clear and self._needs_reset,
                "nested": {"integer": np.uint8(2), "float": np.float64(0.5)}}

    def observation(self):
        self._states = np.full((4, 84, 84), (self._steps % 100) / 100, dtype=np.float32)
        self._rgb = np.full((2, 2, 3), self._steps % 256, dtype=np.uint8)
        return self._states.copy()

    def reset(self, seed=None):
        self.reset_calls += 1
        self._seed, self._steps, self._score, self._current_x = seed, 0, 0, 40
        self._needs_reset = False
        obs, info = self.observation(), self.info()
        if self.on_reset:
            self.on_reset(self, obs, info)
        return obs, info

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Synthetic environment requires reset")
        self.calls += 1
        self.seen_actions.append(action)
        self._steps += 1
        self._current_x += 10
        self._needs_reset = self._steps >= self.length
        obs, info = self.observation(), self.info()
        info["raw_reward"] = 1.0
        info["reward_components"] = {"native": 0.1, "terminal": 0.0}
        reward, terminated, truncated = 0.1, self._needs_reset and not self.is_truncated, self._needs_reset and self.is_truncated
        if self.on_step:
            result = self.on_step(self, obs, reward, terminated, truncated, info)
            if result is not None:
                return result
        return obs, reward, terminated, truncated, info

    def close(self):
        self._closed = True


def config(output, **kwargs):
    return search.SearchConfig(output=Path(output).resolve(), budget=20, horizon=8,
                               max_cells=8, max_prefix=10, max_archive_decisions=40,
                               curve_every=2, **kwargs)


class OwnPrefixTests(unittest.TestCase):
    def run_fake(self, directory, env, *, values=None, clock=None):
        cfg = replace(config(Path(directory) / "run"), **(values or {}))
        engine = search.Search(cfg, lambda: env, environment_kind="synthetic_fixture", clock=clock or Clock())
        result = engine.run()
        return result, engine

    def test_success_requires_full_cold_replay_and_charges_it(self):
        with tempfile.TemporaryDirectory() as directory:
            env = SyntheticEnv(length=3)
            result, engine = self.run_fake(directory, env, values={"budget": 6})
            self.assertEqual(result["status"], "trajectory_verified")
            self.assertEqual(result["counts"], {"exploration": 3, "prefix_replay": 0,
                "trajectory_verification": 3, "total_decisions": 6, "completed_decisions": 6,
                "issued_calls_without_complete_transition_record": 0})
            self.assertEqual(env.calls, 6)
            self.assertEqual(env.reset_calls, 3)
            self.assertEqual(result["matched_replay_transitions"]["trajectory_verification"], 3)
            self.assertFalse(result["network_policy_validated"])
            self.assertFalse(result["model_created"])
            self.assertFalse(result["project_complete"])
            self.assertEqual(list(engine.output.glob("*.pt")), [])
            candidate = json.loads((engine.output / "candidate-trajectory.json").read_text())
            replay = [json.loads(line) for line in (engine.output / "trajectory-verification.jsonl").read_text().splitlines()]
            self.assertEqual(candidate["reset"], replay[0]["snapshot"])
            self.assertEqual([item["transition"] for item in candidate["steps"]],
                             [item["transition"] for item in replay[1:]])
            self.assertIn("_score", replay[0]["snapshot"]["wrapper_fields"])
            for name, proof in result["artifacts"].items():
                self.assertEqual(search.file_hash(engine.output / name), proof["sha256"])

    def test_candidate_without_verification_budget_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            env = SyntheticEnv(length=3)
            result, engine = self.run_fake(directory, env, values={"budget": 5})
            self.assertEqual(result["status"], "candidate_unverified_budget")
            self.assertTrue(result["unverified_candidate"])
            self.assertFalse(result["trajectory_verified"])
            self.assertEqual(env.calls, 3)
            self.assertTrue((engine.output / "candidate-trajectory.json").is_file())
            self.assertFalse((engine.output / "trajectory-verification.jsonl").exists())

    def test_death_and_truncation_never_count_as_clear(self):
        for clear, truncated in [(False, False), (True, True)]:
            with self.subTest(clear=clear, truncated=truncated), tempfile.TemporaryDirectory() as directory:
                env = SyntheticEnv(length=2, clear=clear, truncated=truncated)
                result, _ = self.run_fake(directory, env, values={"budget": 11})
                self.assertEqual(result["status"], "budget_exhausted")
                self.assertEqual(env.calls, 11)
                self.assertFalse(result["trajectory_verified"])
                self.assertEqual(result["terminal_clear_candidates"], 0)

    def test_prefix_replay_and_exploration_share_the_exact_budget(self):
        def choose_longest(archive, rng):
            key, entry = max(archive.entries.items(), key=lambda item: len(item[1].prefix))
            entry.visits += 1
            return key, entry
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(search.Archive, "choose", choose_longest):
            env = SyntheticEnv(length=100, clear=False)
            result, engine = self.run_fake(directory, env, values={"horizon": 1, "budget": 10})
            self.assertEqual(result["counts"]["exploration"], 4)
            self.assertEqual(result["counts"]["prefix_replay"], 6)
            self.assertEqual(env.calls, 10)
            self.assertEqual(result["matched_replay_transitions"]["prefix_replay"], 6)
            with (engine.output / "search-curve.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(int(rows[-1]["total_decisions"]), 10)
            self.assertEqual(rows[-1]["status"], "budget_exhausted")
            self.assertTrue(all(0 <= action < 7 for action in env.seen_actions))

    def test_successful_prefix_uses_all_three_accounting_phases(self):
        def choose_longest(archive, rng):
            return max(archive.entries.items(), key=lambda item: len(item[1].prefix))
        for budget, expected_status in [(9, "trajectory_verified"), (8, "candidate_unverified_budget")]:
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as directory, mock.patch.object(search.Archive, "choose", choose_longest):
                env = SyntheticEnv(length=3)
                result, _ = self.run_fake(directory, env, values={"horizon": 1, "budget": budget})
                self.assertEqual(result["status"], expected_status)
                self.assertEqual(result["counts"]["exploration"], 3)
                self.assertEqual(result["counts"]["prefix_replay"], 3)
                self.assertEqual(result["counts"]["trajectory_verification"], 3 if budget == 9 else 0)
                self.assertEqual(env.calls, 9 if budget == 9 else 6)

    def test_maximum_prefix_limits_actual_search_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            result, engine = self.run_fake(directory, SyntheticEnv(length=100, clear=False),
                values={"budget": 30, "max_prefix": 3, "horizon": 8})
            self.assertEqual(result["status"], "budget_exhausted")
            events = [json.loads(line) for line in (engine.output / "events.jsonl").read_text().splitlines()]
            steps = [item for item in events if item.get("phase") == "exploration"]
            self.assertTrue(all(item["transition"]["after"]["info"]["episode_step"] <= 3 for item in steps))
            self.assertTrue(all(len(item.prefix) < 3 for item in engine.archive.entries.values()))

    def test_full_observation_info_reward_terminal_and_wrapper_drift_are_rejected(self):
        for field in ("observation", "nested_info", "reward", "terminated", "wrapper_score", "rgb"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                def drift(env, obs, reward, terminated, truncated, info):
                    if env.reset_calls >= 3:
                        if field == "observation": obs[0, 0, 0] += np.float32(0.01)
                        if field == "nested_info": info["nested"]["integer"] = 3
                        if field == "reward": reward += 0.25
                        if field == "terminated": terminated = False
                        if field == "wrapper_score": env._score = 999
                        if field == "rgb": env._rgb[0, 0, 0] = 99
                    return obs, reward, terminated, truncated, info
                result, engine = self.run_fake(directory, SyntheticEnv(length=1, on_step=drift))
                self.assertEqual(result["status"], "evidence_mismatch")
                self.assertFalse(result["trajectory_verified"])
                self.assertTrue(result["unverified_candidate"])
                events = [json.loads(line) for line in (engine.output / "events.jsonl").read_text().splitlines()]
                mismatches = [item for item in events if item.get("matched") is False]
                self.assertEqual(len(mismatches), 1)
                self.assertIn("actual_transition", mismatches[0])

    def test_changed_cold_reset_is_rejected_before_next_action(self):
        def drift(env, obs, info):
            if env.reset_calls == 3:
                info["time"] = 299
        with tempfile.TemporaryDirectory() as directory:
            env = SyntheticEnv(length=1, on_reset=drift)
            result, _ = self.run_fake(directory, env)
            self.assertEqual(result["status"], "evidence_mismatch")
            self.assertEqual(env.calls, 1)
            self.assertFalse(result["trajectory_verified"])

    def test_last_verification_step_crossing_deadline_is_not_promoted(self):
        clock = Clock()
        def overrun(env, *args):
            if env.calls == 2: clock.value = 2.0
        with tempfile.TemporaryDirectory() as directory:
            env = SyntheticEnv(length=1, on_step=overrun)
            result, engine = self.run_fake(directory, env, values={"budget": 2,
                "deadline": "1970-01-01T00:00:01+00:00"}, clock=clock)
            self.assertEqual(result["status"], "deadline_reached")
            self.assertFalse(result["trajectory_verified"])
            self.assertEqual(env.calls, 2)
            self.assertEqual(len((engine.output / "trajectory-verification.jsonl").read_text().splitlines()), 2)

    def test_last_verification_step_stop_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "STOP"
            def request_stop(env, *args):
                if env.calls == 2: stop.touch()
            result, _ = self.run_fake(directory, SyntheticEnv(length=1, on_step=request_stop),
                                      values={"stop_file": stop, "budget": 2})
            self.assertEqual(result["status"], "stop_requested")
            self.assertFalse(result["trajectory_verified"])

    def test_preexisting_stop_or_expired_deadline_does_not_construct_environment(self):
        for mode in ("stop", "deadline"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                stop = Path(directory) / "STOP"
                if mode == "stop": stop.touch()
                cfg = replace(config(Path(directory) / "run"), stop_file=stop,
                              deadline="1970-01-01T00:00:00+00:00" if mode == "deadline" else None)
                factory = mock.Mock(side_effect=AssertionError("Must not create a game"))
                result = search.Search(cfg, factory, environment_kind="synthetic_fixture", clock=Clock()).run()
                factory.assert_not_called()
                self.assertEqual(result["counts"]["total_decisions"], 0)
                self.assertEqual(result["archive_cells"], 0)

    def test_step_exception_is_charged_but_not_claimed_recorded(self):
        def failure(env, *args):
            raise RuntimeError("Synthetic failure after interaction")
        with tempfile.TemporaryDirectory() as directory:
            env = SyntheticEnv(on_step=failure)
            result, _ = self.run_fake(directory, env)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["counts"]["total_decisions"], 1)
            self.assertEqual(result["counts"]["issued_calls_without_complete_transition_record"], 1)
            self.assertTrue(env._closed)

    def test_invalid_observation_and_changed_action_set_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            env = SyntheticEnv()
            env.actions = tuple(reversed(env.actions))
            result, _ = self.run_fake(directory, env)
            self.assertEqual(result["status"], "error")
            self.assertEqual(env.calls, 0)
        with tempfile.TemporaryDirectory() as directory:
            def invalid(env, obs, reward, terminated, truncated, info):
                obs[0, 0, 0] = np.nan
            result, _ = self.run_fake(directory, SyntheticEnv(on_step=invalid))
            self.assertEqual(result["status"], "error")
            self.assertFalse(result["trajectory_verified"])

    def test_fresh_output_never_overwrites_an_old_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"; output.mkdir()
            old = output / "manifest.json"; old.write_text("old evidence")
            with self.assertRaises(FileExistsError):
                search.Search(config(output), lambda: SyntheticEnv(), environment_kind="synthetic_fixture")
            self.assertEqual(old.read_text(), "old evidence")

    def test_seed_reproduces_random_actions_and_search_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            environments = [SyntheticEnv(length=100, clear=False), SyntheticEnv(length=100, clear=False)]
            results = []
            for index, env in enumerate(environments):
                cfg = replace(config(Path(directory) / str(index)), budget=30, horizon=3)
                results.append(search.Search(cfg, lambda env=env: env,
                    environment_kind="synthetic_fixture", clock=Clock()).run())
            self.assertEqual(environments[0].seen_actions, environments[1].seen_actions)
            self.assertEqual(results[0]["counts"], results[1]["counts"])

    def test_archive_spatial_diversity_survives_high_reward_visual_flood(self):
        for cell_limit, prefix_limit in [(6, 100), (100, 10)]:
            with self.subTest(cell_limit=cell_limit, prefix_limit=prefix_limit):
                archive = search.Archive(cell_limit, prefix_limit)
                archive.entries["root"] = search.Entry((), 0, 0)
                prefix = ({"synthetic": 1}, {"synthetic": 2})
                obs = np.zeros((4, 84, 84), dtype=np.float32)
                for x in (40, 120):
                    archive.add(obs, {"game_area": 4, "x_pos": x, "y_pos": 80}, prefix, -100, -100)
                for index in range(40):
                    picture = obs.copy()
                    picture[-1, 0, 0] = (index % 8) / 7
                    picture[-1, 0, 12] = (index // 8) / 7
                    archive.add(picture, {"game_area": 4, "x_pos": 700, "y_pos": 80},
                                prefix, 1000 + index, 1000 + index)
                    self.assertLessEqual(len(archive.entries), cell_limit)
                    self.assertLessEqual(archive.stored_decisions, prefix_limit)
                bins = {entry.spatial_bin for entry in archive.entries.values()}
                self.assertIn((4, 40 // 32, 80 // 16), bins)
                self.assertIn((4, 120 // 32, 80 // 16), bins)
                self.assertIn((4, 700 // 32, 80 // 16), bins)

    def test_cli_rejects_unreviewed_real_launch_without_importing_nes(self):
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory) / "review.json"; review.write_text('{}')
            output = Path(directory) / "never-created"
            with self.assertRaises(SystemExit) as error, mock.patch("sys.stderr"):
                search.main(["--output", str(output), "--deadline", "2099-01-01T00:00:00+00:00",
                             "--launch-review", str(review)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_cli_approved_but_stopped_is_a_real_no_game_path(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory).resolve()
            review = folder / "review.json"
            review.write_text(json.dumps({"decision": "approved_own_prefix_search",
                "entropy01_complete_failure_reviewed": True, "all_candidate_histories_reviewed": True}))
            stop = folder / "STOP"; stop.touch()
            original_import = builtins.__import__
            imported_games = []
            def guarded_import(name, *args, **kwargs):
                if name.startswith(("mario_rl", "nes_py", "gym_super_mario_bros")):
                    imported_games.append(name)
                    raise AssertionError("No game may be imported in this stopped CLI test")
                return original_import(name, *args, **kwargs)
            with mock.patch("builtins.__import__", guarded_import), mock.patch("sys.stdout"):
                code = search.main(["--output", str(folder / "run"), "--deadline", "2099-01-01T00:00:00+00:00",
                    "--launch-review", str(review), "--stop-file", str(stop)])
            result = json.loads((folder / "run/manifest.json").read_text())
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], "stop_requested")
            self.assertEqual(result["counts"]["total_decisions"], 0)
            self.assertEqual(imported_games, [])
            self.assertEqual((folder / "run/launch-review.json").read_bytes(), review.read_bytes())

    def parallel_fixture(self, folder):
        folder = Path(folder).resolve()
        stop = folder / "STOP"; stop.touch()
        cfg = search.SearchConfig(output=folder / "run", stop_file=stop,
                                  deadline="2099-01-01T00:00:00+00:00")
        settings = asdict(cfg)
        settings["output"], settings["stop_file"] = str(cfg.output), str(stop)
        old = folder / "original-registration.json"
        old.write_text('{"fixture":"superseded, never launched"}')
        amendment = folder / "registration-v2.json"
        document = {"format_version": 2, "decision_mode": "parallel_after_two_original_recipe_failures",
            "supersedes": {"path": str(old), "sha256": search.file_hash(old)},
            "script": {"path": str(Path(search.__file__).resolve()), "sha256": search.file_hash(search.__file__)},
            "config": settings,
            "comparison_status": {"entropy01": "still_running_at_launch_review_not_judged_failed"}}
        review = {"decision": "approved_parallel_own_prefix_search",
            "original_seed42_and_seed43_complete_failures_reviewed": True,
            "entropy01_still_running": True, "all_candidate_histories_reviewed": True,
            "free_global_job_slot_verified": True}
        review_path = folder / "review.json"
        def save():
            amendment.write_text(json.dumps(document))
            review["amendment_registration"] = {"path": str(amendment), "sha256": search.file_hash(amendment)}
            review_path.write_text(json.dumps(review))
        save()
        command = ["--output", str(cfg.output), "--deadline", cfg.deadline,
                   "--launch-review", str(review_path), "--stop-file", str(stop)]
        return cfg, document, review, review_path, amendment, save, command

    def test_parallel_cli_valid_amendment_stops_without_game_or_failed_entropy_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, document, review, review_path, amendment, _, command = self.parallel_fixture(directory)
            original_import = builtins.__import__
            imported_games = []
            def guarded_import(name, *args, **kwargs):
                if name.startswith(("mario_rl", "nes_py", "gym_super_mario_bros")):
                    imported_games.append(name)
                    raise AssertionError("Parallel stopped CLI must not import a game")
                return original_import(name, *args, **kwargs)
            with mock.patch("builtins.__import__", guarded_import), mock.patch("sys.stdout"):
                code = search.main(command)
            result = json.loads((cfg.output / "manifest.json").read_text())
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], "stop_requested")
            self.assertEqual(result["counts"]["total_decisions"], 0)
            self.assertEqual(imported_games, [])
            self.assertFalse(result["environment_created"])
            self.assertNotIn("entropy01_complete_failure_reviewed", review)
            self.assertEqual((cfg.output / "launch-amendment-registration.json").read_bytes(), amendment.read_bytes())
            self.assertEqual((cfg.output / "superseded-registration.json").read_bytes(), Path(document["supersedes"]["path"]).read_bytes())
            self.assertEqual((cfg.output / "launch-review.json").read_bytes(), review_path.read_bytes())

    def test_parallel_cli_requires_each_live_review_and_rejects_false_failure_claim(self):
        fields = ["original_seed42_and_seed43_complete_failures_reviewed", "entropy01_still_running",
                  "all_candidate_histories_reviewed", "free_global_job_slot_verified", "entropy01_complete_failure_reviewed"]
        for field in fields:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                cfg, _, review, _, _, save, command = self.parallel_fixture(directory)
                review[field] = field == "entropy01_complete_failure_reviewed"
                save()
                with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                    search.main(command)
                self.assertFalse(cfg.output.exists())

    def test_parallel_cli_rejects_missing_or_changed_amendment_bytes(self):
        for mode in ("missing", "changed", "reference_missing"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                cfg, _, review, review_path, amendment, _, command = self.parallel_fixture(directory)
                if mode == "missing": amendment.unlink()
                elif mode == "changed": amendment.write_bytes(amendment.read_bytes() + b" ")
                else:
                    del review["amendment_registration"]
                    review_path.write_text(json.dumps(review))
                with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                    search.main(command)
                self.assertFalse(cfg.output.exists())

    def test_parallel_cli_binds_version_mode_status_script_and_superseded_bytes(self):
        for mode in ("version", "mode", "comparison_status", "script_hash", "other_script", "supersedes"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                cfg, document, _, _, _, save, command = self.parallel_fixture(directory)
                if mode == "version": document["format_version"] = 1
                if mode == "mode": document["decision_mode"] = "old_failure_gate"
                if mode == "comparison_status": document["comparison_status"]["entropy01"] = "failed"
                if mode == "script_hash": document["script"]["sha256"] = "0" * 64
                if mode == "other_script":
                    other = cfg.output.parent / "same-bytes-other-script.py"
                    other.write_bytes(Path(search.__file__).read_bytes())
                    document["script"]["path"] = str(other)
                if mode == "supersedes": Path(document["supersedes"]["path"]).write_text("changed")
                save()
                with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                    search.main(command)
                self.assertFalse(cfg.output.exists())

    def test_parallel_cli_binds_actual_output_deadline_and_all_search_parameters(self):
        for field, value in [("output", "/other/output"), ("deadline", "2098-01-01T00:00:00+00:00"),
                             ("budget", 600_000), ("seed", 43), ("env_seed", 1), ("horizon", 64),
                             ("max_cells", 128), ("max_prefix", 2000), ("max_archive_decisions", 50000),
                             ("curve_every", 128), ("max_hours", 2.0), ("stop_file", None)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                cfg, document, _, _, _, save, command = self.parallel_fixture(directory)
                document["config"][field] = value
                save()
                with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                    search.main(command)
                self.assertFalse(cfg.output.exists())

    def test_parallel_optional_prior_source_archives_are_bound_and_retained(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                cfg, document, _, _, _, save, command = self.parallel_fixture(directory)
                old = {}
                expected = {}
                for name, key, target in [("superseded_script_archive", "script", "superseded-search-script.py"),
                                          ("superseded_tests_archive", "synthetic_tests", "superseded-search-tests.py")]:
                    source = cfg.output.parent / name
                    data = f"# Synthetic original {key}; not executed\n".encode()
                    source.write_bytes(data)
                    old[key] = {"sha256": search.file_hash(source)}
                    document[name] = {"path": str(source), "sha256": search.file_hash(source)}
                    expected[target] = data
                if corrupt:
                    old["script"]["sha256"] = "0" * 64
                old_path = Path(document["supersedes"]["path"])
                old_path.write_text(json.dumps(old))
                document["supersedes"]["sha256"] = search.file_hash(old_path)
                save()
                if corrupt:
                    with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                        search.main(command)
                    self.assertFalse(cfg.output.exists())
                else:
                    with mock.patch("sys.stdout"):
                        self.assertEqual(search.main(command), 2)
                    for name, data in expected.items():
                        self.assertEqual((cfg.output / name).read_bytes(), data)

    def test_parallel_explicit_review_deadline_cannot_disagree(self):
        for field in ("deadline", "absolute_deadline"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                cfg, _, review, _, _, save, command = self.parallel_fixture(directory)
                review[field] = "2098-01-01T00:00:00+00:00"
                save()
                with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                    search.main(command)
                self.assertFalse(cfg.output.exists())

    def test_parallel_empty_absolute_deadline_cannot_remove_round_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, document, _, _, _, save, command = self.parallel_fixture(directory)
            document["config"]["deadline"] = ""
            command[command.index("--deadline") + 1] = ""
            save()
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                search.main(command)
            self.assertFalse(cfg.output.exists())


if __name__ == "__main__":
    unittest.main()
