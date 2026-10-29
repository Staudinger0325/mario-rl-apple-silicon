"""Modern episode semantics around the original NES Gym environment.

Observations match the pretrained PPO: RGB -> grayscale -> 84x84 -> /255,
repeat an action for four environment steps, max-pool the last two preprocessed
frames, and append that one frame to a rolling stack of four decision frames.
Unlike upstream CustomSkipFrame, this wrapper never resets inside step().
The NES backend may skip additional animation frames inside one environment step.
"""
from numbers import Integral

import cv2
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np


REWARD_MODES = ("native", "progress_terminal", "progress_score", "full")


class _SeededJoypadSpace(JoypadSpace):
    # nes-py 8.2.1 JoypadSpace.reset omits seed/options forwarding.
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def process_frame(frame: np.ndarray) -> np.ndarray:
    # Resize before converting to float, as in the checkpoint's training code.
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return (cv2.resize(gray, (84, 84)) / 255.0).astype(np.float32)


class MarioEnv:
    """Seven actions, explicit episode boundaries, and auditable reward modes.

    ``native`` uses the original reward divided by 10; ``progress_terminal``
    adds +/-5 for an actual success/death; ``progress_score`` adds only
    score_delta/400; ``full`` adds both score and extra terminal rewards.
    The three components are recorded even when a mode does not use all of them.
    Artificial maze termination is independently controlled by castle_path_rules.
    Only full mode applies its extra legacy castle reward adjustment. Disable
    those rules for a controlled reward ablation and unmodified game endings.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, world=1, stage=1, seed=0, max_steps=5000, *,
                 reward_mode="full", castle_path_rules=True, _base_env=None):
        if (isinstance(world, (bool, np.bool_)) or not isinstance(world, Integral)
                or isinstance(stage, (bool, np.bool_)) or not isinstance(stage, Integral)
                or not 1 <= world <= 8 or not 1 <= stage <= 4):
            raise ValueError("world must be 1..8 and stage must be 1..4")
        if max_steps is not None and (isinstance(max_steps, (bool, np.bool_))
                                     or not isinstance(max_steps, Integral) or max_steps <= 0):
            raise ValueError("max_steps must be a positive integer or None")
        if not isinstance(reward_mode, str) or reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
        if not isinstance(castle_path_rules, (bool, np.bool_)):
            raise ValueError("castle_path_rules must be a boolean")
        self.world, self.stage = int(world), int(stage)
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.reward_mode = reward_mode
        self.castle_path_rules = bool(castle_path_rules)
        self._seed = seed
        self._env = _base_env if _base_env is not None else _SeededJoypadSpace(
            gym_super_mario_bros.make(f"SuperMarioBros-{self.world}-{self.stage}-v0"),
            SIMPLE_MOVEMENT,
        )
        self.action_space = self._env.action_space
        self.observation_space = gym.spaces.Box(0.0, 1.0, (4, 84, 84), dtype=np.float32)
        self.actions = tuple(tuple(buttons) for buttons in SIMPLE_MOVEMENT)
        self._states = np.zeros((4, 84, 84), dtype=np.float32)
        self._rgb = None
        self._needs_reset = True
        self._closed = False
        self._steps = 0
        self._score = 0
        self._current_x = 40

    def reset(self, seed=None, options=None):
        if self._closed:
            raise RuntimeError("This environment is closed.")
        if seed is not None:
            self._seed = seed
        result = self._env.reset(seed=self._seed, options=options)
        if isinstance(result, tuple):
            frame, info = result
        else:
            frame = result
            info = self._env.unwrapped._get_info()
        self.action_space.seed(self._seed)
        self._rgb = frame.copy()
        processed = process_frame(frame)
        self._states[:] = processed
        self._steps = 0
        self._score = int(info.get("score", 0))
        self._current_x = int(info.get("x_pos", 40))
        self._needs_reset = False
        info = dict(info)
        info["episode_step"] = 0
        self._add_configuration_info(info)
        return self._states.copy(), info

    def _add_configuration_info(self, info):
        """Expose the reward contract and current area without changing game RAM."""
        info["reward_mode"] = self.reward_mode
        info["castle_path_rules"] = self.castle_path_rules
        area = getattr(self._env.unwrapped, "_area", None)
        info["game_area"] = int(area) if area is not None else None

    def _wrong_castle_path(self, info):
        # Retain the upstream task-specific maze termination conditions.
        x, y = info["x_pos"], info["y_pos"]
        if (self.world, self.stage) == (7, 4):
            return (
                (506 <= x <= 832 and y > 127)
                or (832 < x <= 1064 and y < 80)
                or (1113 < x <= 1464 and y < 191)
                or (1579 < x <= 1943 and y < 191)
                or (1946 < x <= 1964 and y >= 191)
                or (1984 < x <= 2060 and (y >= 191 or y < 127))
                or (2114 < x < 2440 and y < 191)
                or x < self._current_x - 500
            )
        if (self.world, self.stage) == (4, 4):
            return (x <= 1500 and y < 127) or (1588 <= x < 2380 and y >= 127)
        return False

    def step(self, action):
        if self._closed or self._needs_reset:
            raise RuntimeError("Call reset() before stepping a new or finished episode.")
        if not isinstance(action, Integral) or not self.action_space.contains(int(action)):
            raise ValueError(f"action must be an integer in 0..{self.action_space.n - 1}")
        total_reward = 0.0
        raw_reward = 0.0
        components = {"native": 0.0, "score": 0.0, "terminal": 0.0, "castle_rule": 0.0}
        applied = {key: 0.0 for key in components}
        castle_rule_triggered = False
        frames = []
        terminated = truncated = False
        for _ in range(4):
            result = self._env.step(int(action))
            if len(result) == 5:
                frame, reward, terminated, truncated, info = result
            else:
                frame, reward, done, info = result
                truncated = bool(done and info.get("TimeLimit.truncated", False))
                terminated = bool(done and not truncated)
            info = dict(info)
            self._rgb = frame.copy()
            frames.append(process_frame(frame))
            raw_reward += float(reward)
            native = float(reward) / 10.0
            score = (int(info["score"]) - self._score) / 400.0
            self._score = int(info["score"])
            terminal = (5.0 if info.get("flag_get", False) else -5.0) if terminated else 0.0
            castle_rule = 0.0
            if self.castle_path_rules and self._wrong_castle_path(info):
                # Preserve the old full-mode reward without disguising it as
                # an ordinary game death or modifying the success flag.
                castle_rule = (-5.0 - native - score - terminal) if self.world == 4 else -5.0
                terminated = True
                castle_rule_triggered = True
                info["termination_reason"] = "wrong_castle_path"
            self._current_x = int(info["x_pos"])
            frame_components = dict(native=native, score=score, terminal=terminal, castle_rule=castle_rule)
            for key, value in frame_components.items():
                components[key] += value
            applied["native"] += native
            if self.reward_mode in ("progress_terminal", "full"):
                applied["terminal"] += terminal
            if self.reward_mode in ("progress_score", "full"):
                applied["score"] += score
            if self.reward_mode == "full":
                applied["castle_rule"] += castle_rule
            if terminated or truncated:
                break
        total_reward = sum(applied.values())
        self._states[:-1] = self._states[1:]
        self._states[-1] = np.maximum.reduce(frames[-2:])
        self._steps += 1
        if self.max_steps is not None and self._steps >= self.max_steps and not terminated:
            truncated = True
            info["TimeLimit.truncated"] = True
        self._needs_reset = bool(terminated or truncated)
        info["episode_step"] = self._steps
        info["frames_advanced"] = len(frames)
        info["raw_reward"] = raw_reward
        info.update({f"reward_{key}": float(value) for key, value in components.items()})
        info["reward_components"] = components
        info["applied_reward_components"] = applied
        info["shaped_reward"] = float(total_reward)
        info["castle_path_rule_triggered"] = castle_rule_triggered
        self._add_configuration_info(info)
        return self._states.copy(), float(total_reward), bool(terminated), bool(truncated), info

    def render(self):
        if self._rgb is None:
            raise RuntimeError("Call reset() before render().")
        return self._rgb.copy()

    def close(self):
        if not self._closed:
            self._env.close()
            self._closed = True


def make_env(world=1, stage=1, seed=0, max_steps=5000, *,
             reward_mode="full", castle_path_rules=True) -> MarioEnv:
    return MarioEnv(world=world, stage=stage, seed=seed, max_steps=max_steps,
                    reward_mode=reward_mode, castle_path_rules=castle_path_rules)
