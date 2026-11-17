"""Shared pieces: config loading, env construction, the residual wrapper, callbacks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

from orca_sim import OrcaHandGrasp, ScriptedGraspPolicy

RANGE_KEYS = ("object_mass", "object_friction")  # YAML [lo, hi] -> (lo, hi) range


# ---------------------------------------------------------------- config
def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def env_kwargs_from(cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(cfg["env"])
    for key in RANGE_KEYS:
        if isinstance(kwargs.get(key), list):
            kwargs[key] = tuple(kwargs[key])
    return kwargs


# ---------------------------------------------------------------- residual action wrapper
class ResidualActionWrapper(gym.Wrapper):
    """The policy outputs a correction in [-1, 1] per joint; the hand gets

        joint_target = scripted_grasp(t) + scale * correction

    With correction = 0 this is exactly the scripted grasp, so training starts from a
    policy that already works and only has to learn improvements. `scale` (rad) limits
    how far the policy can deviate from the script.
    """

    def __init__(self, env: OrcaHandGrasp, scale: float = 0.3, close_steps: int = 80) -> None:
        super().__init__(env)
        self.scale = scale
        self.scripted = ScriptedGraspPolicy(close_steps=close_steps)
        n = env.action_space.shape[0]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)
        self._t = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.scripted.reset(self.env)
        self._t = 0
        return obs, info

    def step(self, action):
        correction = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target = self.scripted.action(self._t) + self.scale * correction
        self._t += 1
        return self.env.step(target)  # the env clips to the joint limits


def make_env(cfg: dict[str, Any], render_mode: str | None = None) -> gym.Env:
    env = OrcaHandGrasp(render_mode=render_mode, **env_kwargs_from(cfg))
    res = cfg.get("residual")
    if res:
        env = ResidualActionWrapper(env, scale=res["scale"], close_steps=res.get("close_steps", 80))
    else:
        raise NotImplementedError("only the residual setup is implemented so far")
    return env


# ---------------------------------------------------------------- callbacks
class RewardTermsLogger(BaseCallback):
    """Logs the per-episode sum of every reward term (weighted or not) to TensorBoard
    under terms/<name>. This is how you see WHAT the agent is optimizing."""

    def _on_training_start(self) -> None:
        self._sums = [defaultdict(float) for _ in range(self.training_env.num_envs)]
        self._finished: dict[str, list[float]] = defaultdict(list)

    def _on_step(self) -> bool:
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            for k, v in info.get("reward_terms", {}).items():
                self._sums[i][k] += v
            if done:
                for k, v in self._sums[i].items():
                    self._finished[k].append(v)
                self._sums[i] = defaultdict(float)
        return True

    def _on_rollout_end(self) -> None:
        for k, values in self._finished.items():
            if values:
                self.logger.record(f"terms/{k}", float(np.mean(values)))
        self._finished = defaultdict(list)


class RewardCurriculum(BaseCallback):
    """Switch on reward terms in stages, once the training success rate is high enough.

    Config example:
        curriculum:
          - {min_success: 0.8, weights: {grip_force: -0.02}}
          - {min_success: 0.8, weights: {grip_force: -0.05, table_contact: -0.1}}
    Each stage waits until the success rate (last 100 episodes) reaches min_success,
    then calls env.set_reward_weights(weights) on every training env.
    """

    def __init__(self, stages: list[dict[str, Any]], eval_env=None) -> None:
        super().__init__()
        self.stages = stages
        self.eval_env = eval_env
        self.stage = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("curriculum/stage", self.stage)
        buf = self.model.ep_success_buffer
        if self.stage >= len(self.stages) or len(buf) < 50:
            return
        stage = self.stages[self.stage]
        if np.mean(buf) >= stage["min_success"]:
            self.training_env.env_method("set_reward_weights", stage["weights"])
            if self.eval_env is not None:
                self.eval_env.env_method("set_reward_weights", stage["weights"])
            self.stage += 1
            self.model.ep_success_buffer.clear()  # judge the next stage on fresh episodes
            print(f"[curriculum] step {self.num_timesteps:,}: stage {self.stage} -> {stage['weights']}")
