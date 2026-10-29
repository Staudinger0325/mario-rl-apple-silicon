"""CPU-only serial or spawn-process game sampling for macOS PPO training."""
from __future__ import annotations

import multiprocessing as mp
import traceback

import numpy as np


def _worker(connection, kwargs):
    from .env import make_env
    env = None
    try:
        env = make_env(**kwargs)
        connection.send((True, "ready"))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                result = env.reset(seed=payload)
            elif command == "step":
                result = env.step(int(payload))
            else:
                raise ValueError(f"Unknown worker command: {command}")
            connection.send((True, result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send((False, traceback.format_exc()))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class VectorEnv:
    """Return final observations without autoreset; reset only chosen games."""
    def __init__(self, num_envs=4, world=1, stage=1, seed=0,
                 reward_mode="full", castle_path_rules=False,
                 backend="serial", max_steps=5000, timeout=60):
        if backend not in {"serial", "process"} or num_envs < 1:
            raise ValueError("backend must be serial/process and num_envs must be positive")
        self.num_envs = num_envs
        self.backend = backend
        self.seed = seed
        self.timeout = timeout
        self.envs = []
        self.connections = []
        self.processes = []
        self.closed = False
        configs = [dict(world=world, stage=stage, seed=seed + index,
                        reward_mode=reward_mode, castle_path_rules=castle_path_rules,
                        max_steps=max_steps) for index in range(num_envs)]
        try:
            if backend == "serial":
                from .env import make_env
                for config in configs:
                    self.envs.append(make_env(**config))
            else:
                context = mp.get_context("spawn")
                for config in configs:
                    parent, child = context.Pipe()
                    process = context.Process(target=_worker, args=(child, config), daemon=True)
                    process.start()
                    child.close()
                    self.connections.append(parent)
                    self.processes.append(process)
                for index in range(num_envs):
                    self._receive(index)
        except BaseException:
            self.close()
            raise

    def _receive(self, index):
        connection = self.connections[index]
        if not connection.poll(self.timeout):
            raise TimeoutError(f"Environment worker {index} did not reply within {self.timeout}s")
        try:
            ok, result = connection.recv()
        except EOFError as error:
            raise RuntimeError(f"Environment worker {index} exited unexpectedly") from error
        if not ok:
            raise RuntimeError(f"Environment worker {index} failed:\n{result}")
        return result

    def reset_at(self, indices):
        if self.closed:
            raise RuntimeError("VectorEnv is closed")
        indices = list(indices)
        if len(set(indices)) != len(indices) or any(not 0 <= i < self.num_envs for i in indices):
            raise ValueError("Reset indices must be unique and in range")
        if self.backend == "serial":
            return {i: self.envs[i].reset(seed=self.seed + i) for i in indices}
        for index in indices:
            self.connections[index].send(("reset", self.seed + index))
        return {index: self._receive(index) for index in indices}

    def reset(self):
        result = self.reset_at(range(self.num_envs))
        return np.stack([result[i][0] for i in range(self.num_envs)]), [result[i][1] for i in range(self.num_envs)]

    def step(self, actions):
        if self.closed:
            raise RuntimeError("VectorEnv is closed")
        if len(actions) != self.num_envs:
            raise ValueError("Exactly one action is required per environment")
        if self.backend == "serial":
            results = [env.step(int(action)) for env, action in zip(self.envs, actions)]
        else:
            for connection, action in zip(self.connections, actions):
                connection.send(("step", int(action)))
            results = [self._receive(i) for i in range(self.num_envs)]
        obs, rewards, terminated, truncated, infos = zip(*results)
        return (np.stack(obs), np.asarray(rewards, dtype=np.float32),
                np.asarray(terminated, dtype=bool), np.asarray(truncated, dtype=bool), list(infos))

    def close(self):
        if self.closed:
            return
        self.closed = True
        for env in self.envs:
            env.close()
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (EOFError, BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for connection in self.connections:
            connection.close()

