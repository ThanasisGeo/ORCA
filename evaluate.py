"""Evaluate a trained run against the scripted baseline, per object.

    python evaluate.py runs/<run>                         # trained policy vs scripted, 100 episodes
    python evaluate.py runs/<run> --objects primitives    # test on objects it may not have trained on
    python evaluate.py runs/<run> --render                # watch in the MuJoCo viewer (5 episodes each)
    python evaluate.py runs/<run> --render --objects primitives --skip-scripted --episodes 10
    MUJOCO_GL=egl python evaluate.py runs/<run> --video   # save videos/<run>.mp4
    python evaluate.py --scripted-only configs/residual_cube.yaml   # baseline only, no model needed

The scripted baseline is the same residual env with a zero correction, i.e. exactly
the hand-coded grasp under the same conditions as the policy.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from common import load_config, make_env


def run_episodes(cfg, n, seed, model=None, vecnorm_path=None, render_mode=None):
    """Returns per-object [successes, episodes], mean return, mean penalty terms, frames."""
    venv = DummyVecEnv([lambda: make_env(cfg, render_mode=render_mode)])
    if model is not None and vecnorm_path and Path(vecnorm_path).exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False
    venv.seed(seed)
    per_object = defaultdict(lambda: [0, 0])
    returns, frames = [], []
    terms = defaultdict(float)
    watching = render_mode == "human"

    def announce():  # which object is in the viewer now
        o = venv.envs[0].unwrapped._object
        print(f"  episode {len(returns) + 1}/{n}: {o['name']:15s} mass {1000 * o['mass']:5.0f} g   "
              f"friction {o['friction']:.2f}   needs m·g/μ = {o['mass'] * 9.81 / o['friction']:.2f} N",
              end="", flush=True)

    try:
        obs = venv.reset()
        ep_ret = 0.0
        if watching:
            announce()
        while len(returns) < n:
            if model is None:
                action = np.zeros((1, venv.action_space.shape[0]), dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=True)
            t0 = time.perf_counter()
            obs, reward, done, infos = venv.step(action)
            if watching:  # play at real speed (1 step = 0.01 s) instead of as fast as possible
                time.sleep(max(0.0, 0.01 - (time.perf_counter() - t0)))
            ep_ret += float(reward[0])
            info = infos[0]
            for k, v in info.get("reward_terms", {}).items():
                terms[k] += v
            if render_mode == "rgb_array":
                frames.append(venv.envs[0].render())
            if done[0]:
                name = info["object"]["name"]
                per_object[name][0] += int(info.get("is_success", False))
                per_object[name][1] += 1
                returns.append(ep_ret)
                ep_ret = 0.0
                if watching:
                    print("   ->", "HELD" if info.get("is_success") else "dropped/slipped", flush=True)
                    if len(returns) < n:  # the env already reset to the next object
                        announce()
    finally:
        venv.close()  # also on Ctrl+C, so the viewer window shuts down cleanly
    return per_object, float(np.mean(returns)), {k: v / n for k, v in terms.items()}, frames


def report(label, per_object, mean_ret, terms):
    s = sum(a for a, _ in per_object.values())
    n = sum(b for _, b in per_object.values())
    print(f"\n{label}: success {s}/{n} = {100 * s / n:.0f}%   mean return {mean_ret:.1f}")
    for name, (a, b) in sorted(per_object.items()):
        print(f"   {name:15s} {a:3d}/{b:<3d} {'#' * round(20 * a / b)}")
    print("   per-episode term sums: " + ", ".join(f"{k}={v:.1f}" for k, v in terms.items()
                                                    if k not in ("reach", "hold", "drop", "success")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="run folder (runs/...), or a config with --scripted-only")
    parser.add_argument("--which", default="best_model", help="best_model or final_model")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--objects", help="override env.objects (set name or comma-separated names)")
    parser.add_argument("--side", choices=["left", "right"], help="override env.side")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--scripted-only", action="store_true")
    parser.add_argument("--skip-scripted", action="store_true", help="only run the trained policy")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    target = Path(args.target)
    cfg = load_config(target if args.scripted_only else target / "config.yaml")
    if args.objects:
        cfg["env"]["objects"] = args.objects.split(",") if "," in args.objects else args.objects
    if args.side:
        cfg["env"]["side"] = args.side
    render_mode = "human" if args.render else ("rgb_array" if args.video else None)
    n = (args.episodes if args.episodes != 100 else 5) if (args.render or args.video) else args.episodes

    if not args.skip_scripted:
        if render_mode == "human":
            print("\nSCRIPTED baseline:")
        report("SCRIPTED baseline", *run_episodes(cfg, n, args.seed, render_mode=render_mode)[:3])
    if args.scripted_only:
        return

    model = PPO.load(target / args.which, device="cpu")
    if render_mode == "human":
        print(f"\nRL policy ({args.which}):")
    per_obj, ret, terms, frames = run_episodes(cfg, n, args.seed, model, target / "vecnormalize.pkl",
                                               render_mode=render_mode)
    report(f"RL policy ({args.which})", per_obj, ret, terms)
    if args.video:
        import imageio
        out = Path("videos") / f"{target.name}.mp4"
        out.parent.mkdir(exist_ok=True)
        imageio.mimsave(out, frames[::2], fps=50)  # sim runs at 100 steps/s -> real time
        print(f"saved {out}")


if __name__ == "__main__":
    main()
