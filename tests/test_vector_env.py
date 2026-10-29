"""Compare serial and spawn workers against the same actual emulator trajectory."""
import unittest

import numpy as np

from mario_rl.vector_env import VectorEnv


class VectorTests(unittest.TestCase):
    def test_process_matches_serial_and_preserves_terminal_observation(self):
        serial = VectorEnv(num_envs=2, backend="serial", max_steps=2, seed=31)
        process = VectorEnv(num_envs=2, backend="process", max_steps=2, seed=31)
        try:
            a, info_a = serial.reset()
            b, info_b = process.reset()
            np.testing.assert_array_equal(a, b)
            self.assertEqual(info_a, info_b)
            for actions in ([1, 4], [2, 3]):
                a = serial.step(actions)
                b = process.step(actions)
                for index in range(4):
                    np.testing.assert_array_equal(a[index], b[index])
                self.assertEqual(a[4], b[4])
            self.assertTrue(a[3].all())
            old_last = b[0][1].copy()
            reset = process.reset_at([0])
            self.assertEqual(set(reset), {0})
            self.assertEqual(reset[0][1]["episode_step"], 0)
            np.testing.assert_array_equal(b[0][1], old_last)
            with self.assertRaises(RuntimeError):
                process.step([0, 0])  # Worker 1 needs an explicit reset.
        finally:
            serial.close()
            process.close()

    def test_validation_and_cleanup(self):
        with self.assertRaises(ValueError):
            VectorEnv(backend="fork")
        env = VectorEnv(num_envs=1)
        try:
            with self.assertRaises(ValueError):
                env.reset_at([0, 0])
            with self.assertRaises(ValueError):
                env.step([])
        finally:
            env.close()
        with self.assertRaises(RuntimeError):
            env.reset()


if __name__ == "__main__":
    unittest.main()
