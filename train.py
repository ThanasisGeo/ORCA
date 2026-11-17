"""Train a residual PPO policy on the ORCA grasp task.

    python train.py configs/residual_cube.yaml
    python train.py configs/residual_cube.yaml --steps 200000 --tag quicktest

Everything for a run goes into runs/<config name>_<timestamp>[_tag]/ :
    config.yaml        exact config used (copy)
    tb/                TensorBoard logs     -> tensorboard --logdir runs
    best_model.zip     best by evaluation success/return
    final_model.zip, vecnormalize.pkl, checkpoints/
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from common import RewardCurriculum, RewardTermsLogger, load_config, make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--steps", type=int, help="override training.total_timesteps")
    parser.add_argument("--n-envs", type=int, help="override training.n_envs")
    parser.add_argument("--seed", type=int, help="override seed")
    parser.add_argument("--tag", default="", help="suffix for the run folder name")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr = cfg["training"]
    steps = args.steps or tr["total_timesteps"]
    n_envs = args.n_envs or tr["n_envs"]
    seed = cfg.get("seed", 0) if args.seed is None else args.seed

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / (f"{Path(args.config).stem}_{stamp}" + (f"_{args.tag}" if args.tag else ""))
    run_dir.mkdir(parents=True)
    shutil.copy(args.config, run_dir / "config.yaml")

    torch.set_num_threads(tr.get("torch_threads", 4))  # for the PPO update; sims run in their own processes

    env = make_vec_env(lambda: make_env(cfg), n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(lambda: make_env(cfg), n_envs=1, seed=seed + 10_000)
    env = VecNormalize(env, norm_obs=tr["normalize_obs"], norm_reward=tr["normalize_reward"],
                       gamma=cfg["ppo"]["gamma"])
    eval_env = VecNormalize(eval_env, norm_obs=tr["normalize_obs"], norm_reward=False, training=False)

    ppo_kwargs = dict(cfg["ppo"])
    policy_kwargs = ppo_kwargs.pop("policy_kwargs", {})
    if "activation_fn" in policy_kwargs:
        policy_kwargs["activation_fn"] = getattr(torch.nn, policy_kwargs["activation_fn"])
    model = PPO("MlpPolicy", env, seed=seed, verbose=1, tensorboard_log=str(run_dir / "tb"),
                policy_kwargs=policy_kwargs, **ppo_kwargs)

    eval_every = max(tr.get("eval_every", 100_000) // n_envs, 1)
    callbacks = [
        RewardTermsLogger(),
        EvalCallback(eval_env, eval_freq=eval_every, n_eval_episodes=tr.get("eval_episodes", 32),
                     best_model_save_path=str(run_dir), deterministic=True),
        CheckpointCallback(save_freq=max(500_000 // n_envs, 1), save_path=str(run_dir / "checkpoints"),
                           save_vecnormalize=True),
    ]
    if cfg.get("curriculum"):
        callbacks.append(RewardCurriculum(cfg["curriculum"], eval_env=eval_env))

    print(f"run: {run_dir} | {steps:,} steps | {n_envs} envs")
    try:
        model.learn(total_timesteps=steps, callback=callbacks, tb_log_name="ppo", progress_bar=True)
    except KeyboardInterrupt:
        print("\nCtrl+C: stopping early and saving what we have.")
    model.save(run_dir / "final_model")
    env.save(str(run_dir / "vecnormalize.pkl"))  # needed by evaluate.py / diagnose.py
    print(f"\ndone. Evaluate with:\n  python evaluate.py {run_dir}\n  python diagnose.py {run_dir}")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
