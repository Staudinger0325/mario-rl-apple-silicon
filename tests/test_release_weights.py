"""Offline integrity and level selection checks for the public local models."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from mario_rl.evaluate import policy_checkpoint, policy_model
from mario_rl.weights import default_manifest_path, ensure_weights, load_manifest, resolve_weights, tensor_digest


class ReleaseWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.path = Path(__file__).resolve().parents[1] / "weights/manifest.json"

    def test_four_real_exports_are_plain_tensors_and_preserve_original_model_hashes(self):
        _, manifest = load_manifest(self.path)
        for record in manifest["models"]:
            path, _ = resolve_weights(1, record["stage"], self.path)
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(len(state), 14)
            self.assertTrue(all(isinstance(x, torch.Tensor) for x in state.values()))
            self.assertEqual(tensor_digest(state), record["model_sha256"])
            self.assertEqual(record["validation"]["wins"], 5)
            self.assertNotEqual(record["sha256"], record["original_checkpoint_sha256"])

    def test_manifest_has_english_metadata_and_relative_model_paths(self):
        value = json.loads(self.path.read_text())
        def visit(node):
            if isinstance(node, str):
                self.assertTrue(node.isascii())
                self.assertNotIn("/Users/", node)
                self.assertNotIn("/home/", node)
            elif isinstance(node, dict):
                for key, item in node.items(): visit(key);visit(item)
            elif isinstance(node, list):
                for item in node:visit(item)
        visit(value)
        self.assertTrue(all(not Path(r["path"]).is_absolute() for r in value["models"]))

    def test_stage_three_defaults_to_local_bc_without_ppo_mislabeling(self):
        path, record = policy_checkpoint("local", None, 1, 3, self.path)
        self.assertEqual(record["method"], "own_search_bc")
        self.assertEqual(record["selected_ppo_decisions"], 0)
        self.assertIn("bc", path.name)
        model = policy_model("local", None, "cpu", 1, 3, self.path)
        self.assertEqual(tensor_digest(model.state_dict()), record["model_sha256"])

    def test_other_world_needs_explicit_checkpoint_but_random_does_not(self):
        with self.assertRaisesRegex(ValueError, "provide --checkpoint"):
            policy_checkpoint("local", None, 2, 1, self.path)
        path = ensure_weights(1, 1, self.path)
        self.assertEqual(policy_checkpoint("local", path, 2, 1)[0], path)
        self.assertEqual(policy_checkpoint("random", None, 8, 4), (None, None))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            policy_checkpoint("random", path)

    def test_missing_local_files_fail_offline_without_a_download(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict("os.environ", {"MARIO_WEIGHTS_DIR": folder}), \
             patch("urllib.request.urlopen", side_effect=AssertionError("No downloads")):
            with self.assertRaises(FileNotFoundError):ensure_weights()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_tampered_file_and_wrong_manifest_method_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "weights"
            shutil.copytree(self.path.parent, target)
            manifest = json.loads((target / "manifest.json").read_text())
            weight = target / manifest["models"][0]["path"]
            with weight.open("ab") as stream:stream.write(b"altered")
            with self.assertRaisesRegex(ValueError, "file checksum"):
                resolve_weights(1, 1, target / "manifest.json")
            manifest["models"][2]["method"] = "ppo"
            (target / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_manifest(target / "manifest.json")

    def test_cli_default_is_local_and_demo_verification_does_not_create_a_game(self):
        from mario_rl.__main__ import main
        from scripts.play_world1 import main as demo
        with patch("sys.argv", ["mario_rl", "evaluate", "--stage", "3"]), patch("mario_rl.evaluate.evaluate") as evaluate:
            main()
        self.assertEqual(evaluate.call_args.args[0].policy, "local")
        with patch("mario_rl.env.make_env", side_effect=AssertionError("Verify must not run NES")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(demo(["--manifest", str(self.path), "--verify"]), 0)
        self.assertFalse(json.loads(output.getvalue())["game_started"])


if __name__ == "__main__":
    unittest.main()
