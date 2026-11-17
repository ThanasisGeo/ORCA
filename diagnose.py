"""Does the grip adapt to mass and friction? Does the hand react when the load changes?

Runs a grid of masses x frictions on ONE fixed object shape (so only mass/friction vary),
for the scripted grasp or a trained run, and writes CSV + plots.

    python diagnose.py --scripted configs/residual_cube_grip.yaml   # -> runs/scripted_<config>/diagnose/
    python diagnose.py runs/<run>                                   # -> runs/<run>/diagnose/
    python diagnose.py runs/<run> --masses 0.02 0.08 0.15 --frictions 0.5 1.0 --episodes 5

Plots:
  success_grid.png       success rate per (mass, friction) cell
  force_vs_required.png  grip force while holding vs. the minimum needed (m*g/mu).
                         Adapting grip = points follow the diagonal band; a constant
                         "squeeze everything" grip = a flat horizontal line.
  timeseries.png         grip force, finger closing command and slip over time for a light/grippy
                         and a heavy/slippery object.
  reaction.png           "if slip then grip" test: the same episode with and without an extra
                         downward pull on the object part-way through the hold. A reactive policy
                         closes harder (closing command goes up) after the pull starts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from common import load_config, make_env  # noqa: E402

FLEX_JOINTS = {"t-mcp", "t-pip", *(f"{f}-{j}" for f in "imrp" for j in ("mcp", "pip"))}


class Policy:
    """Scripted (zero residual) or a trained PPO model with its observation normalization."""

    def __init__(self, cfg, run_dir: Path | None, which: str):
        self.model = None
        if run_dir is not None:
            self.model = PPO.load(run_dir / which, device="cpu")
            self.norm = None
            path = run_dir / "vecnormalize.pkl"
            if path.exists():
                self.norm = VecNormalize.load(str(path), DummyVecEnv([lambda: make_env(cfg)]))
                self.norm.training = False

    def __call__(self, obs: np.ndarray, n_act: int) -> np.ndarray:
        if self.model is None:
            return np.zeros(n_act, dtype=np.float32)
        o = self.norm.normalize_obs(obs[None]) if self.norm is not None else obs[None]
        action, _ = self.model.predict(o, deterministic=True)
        return action[0]


def run_episode(env, policy, mass, friction, seed, pull_newton=0.0, pull_step=None):
    core = env.unwrapped
    core.object_mass, core.object_friction = mass, friction
    obs, _ = env.reset(seed=seed)
    flex = [i for i, n in enumerate(core.actuated_joint_names(strip_side=True)) if n in FLEX_JOINTS]
    log = {"grip": [], "flex_cmd": [], "disp": [], "effort": []}
    n_act = env.action_space.shape[0]
    info = {}
    for t in range(core.max_episode_steps):
        if pull_step is not None and t == pull_step:
            core.data.xfrc_applied[core._obj_body, 2] = -pull_newton  # extra downward load
        obs, _, terminated, truncated, info = env.step(policy(obs, n_act))
        log["grip"].append(info["reward_terms"]["grip_force"])
        log["flex_cmd"].append(float(np.mean(core.data.ctrl[flex])))
        log["disp"].append(info["displacement"])
        log["effort"].append(float(np.mean(np.abs(core.data.actuator_force))))
        if terminated or truncated:
            break
    removal = core.table_removal_step
    hold = slice(removal + 10, len(log["grip"]))  # after the table is gone (skip the first jolt)
    summary = {
        "mass": mass, "friction": friction, "seed": seed,
        "success": int(bool(info.get("is_success"))),
        "steps_survived": len(log["grip"]),
        "required_force": info["required_grip_force"],
        # Averages over the hold phase (table gone). NaN if the object fell before that.
        "grip_hold_mean": float(np.mean(log["grip"][hold])) if len(log["grip"]) > hold.start else np.nan,
        "closing_cmd_hold_mean": float(np.mean(log["flex_cmd"][hold])) if len(log["grip"]) > hold.start else np.nan,
        "effort_mean": float(np.mean(log["effort"])),
        "final_displacement_cm": 100 * log["disp"][-1],
    }
    return summary, log


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("target", nargs="?", help="run folder (runs/...)")
    p.add_argument("--scripted", metavar="CONFIG", help="diagnose the scripted grasp under this config")
    p.add_argument("--which", default="best_model")
    p.add_argument("--object", default="cube_4cm", help="fixed object shape for the grid")
    p.add_argument("--masses", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.15])
    p.add_argument("--frictions", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.2])
    p.add_argument("--episodes", type=int, default=3, help="episodes per grid cell")
    p.add_argument("--jitter", type=float, default=0.003, help="start xy jitter (m) so episodes differ")
    p.add_argument("--pull", type=float, default=0.5, help="extra downward force for the reaction test (N)")
    args = p.parse_args()
    if bool(args.target) == bool(args.scripted):
        p.error("give either a run folder or --scripted CONFIG")

    if args.scripted:
        cfg = load_config(args.scripted)
        run_dir, out = None, Path("runs") / f"scripted_{Path(args.scripted).stem}" / "diagnose"
        label = "scripted"
    else:
        run_dir = Path(args.target)
        cfg = load_config(run_dir / "config.yaml")
        out, label = run_dir / "diagnose", run_dir.name
    cfg["env"]["objects"] = [args.object]
    # Small start-position jitter so the episodes in a cell differ (the policy is deterministic).
    cfg["env"]["object_xy_jitter"] = args.jitter
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg)
    policy = Policy(cfg, run_dir, args.which)
    safety = env.unwrapped.grip_safety_factor
    removal = env.unwrapped.table_removal_step

    # ---------------------------------------------------------------- grid
    rows = []
    for m in args.masses:
        for f in args.frictions:
            for k in range(args.episodes):
                rows.append(run_episode(env, policy, m, f, seed=1000 + k)[0])
    with open(out / "grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    M, F = len(args.masses), len(args.frictions)
    succ = np.zeros((M, F))
    for r in rows:
        succ[args.masses.index(r["mass"]), args.frictions.index(r["friction"])] += r["success"] / args.episodes

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(succ, vmin=0, vmax=1, cmap="viridis", origin="lower")
    ax.set_xticks(range(F), [f"{f:g}" for f in args.frictions])
    ax.set_yticks(range(M), [f"{1000 * m:g}" for m in args.masses])
    ax.set_xlabel("friction")
    ax.set_ylabel("mass (g)")
    for i in range(M):
        for j in range(F):
            ax.text(j, i, f"{succ[i, j]:.0%}", ha="center", va="center", color="w" if succ[i, j] < 0.6 else "k")
    fig.colorbar(im, label="success rate")
    ax.set_title(f"Success: {label}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "success_grid.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- force vs required
    # Two views. (a) DECISION: what the policy commands (closing angle) vs what the object needs.
    # (b) OUTCOME: the contact force that resulted. The outcome also depends on physics (a heavier
    # object presses harder into the fingers even with the same command), so judge adaptation by (a).
    req = np.array([r["required_force"] for r in rows])
    cmd = np.array([r["closing_cmd_hold_mean"] for r in rows])
    force = np.array([r["grip_hold_mean"] for r in rows])
    ok = np.array([r["success"] for r in rows], bool)
    valid = ~np.isnan(cmd)  # objects that fell before/at table removal have no hold phase

    def fit_slope(y):
        v = valid & ~np.isnan(y)
        return np.polyfit(req[v], y[v], 1)[0] if len(set(req[v])) > 1 else np.nan

    slope_cmd, slope_force = fit_slope(cmd), fit_slope(force)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, y, ylab, slope in [
        (axs[0], cmd, "mean finger closing command\nwhile holding (rad)", slope_cmd),
        (axs[1], force, "grip force while holding (N)", slope_force),
    ]:
        ax.scatter(req[ok], y[ok], c="tab:blue", label="held", s=25)
        ax.scatter(req[~ok & valid], y[~ok & valid], c="tab:red", marker="x", label="dropped later", s=30)
        ax.set_xlabel("required grip force m·g/μ (N)")
        ax.set_ylabel(ylab)
        ax.text(0.02, 0.97, f"slope of fit: {slope:.3f}", transform=ax.transAxes, va="top", fontsize=8)
    x = np.linspace(0, req.max() * 1.1, 10)
    axs[1].plot(x, x, "k--", lw=1, label="minimum needed (m·g/μ)")
    axs[1].plot(x, safety * x, ":", c="gray", lw=1, label=f"{safety:g}× (grip penalty starts above)")
    axs[0].set_title("(a) DECISION: does the command follow the need?", fontsize=9)
    axs[1].set_title("(b) OUTCOME: resulting contact force", fontsize=9)
    axs[1].legend(fontsize=8)
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "force_vs_required.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- time series
    cases = {
        f"light {1000 * min(args.masses):g} g, μ={max(args.frictions):g}": (min(args.masses), max(args.frictions)),
        f"heavy {1000 * max(args.masses):g} g, μ={min(args.frictions):g}": (max(args.masses), min(args.frictions)),
    }
    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    for name, (m, f) in cases.items():
        s, log = run_episode(env, policy, m, f, seed=1000)
        t = np.arange(len(log["grip"])) / 100.0
        axs[0].plot(t, log["grip"], label=f"{name} ({'held' if s['success'] else 'dropped'})")
        axs[1].plot(t, log["flex_cmd"])
        axs[2].plot(t, 100 * np.array(log["disp"]))
    for ax, lab in zip(axs, ["grip force (N)", "mean finger closing\ncommand (rad)", "slip since table\nremoval (cm)"]):
        ax.axvline(removal / 100, c="k", ls="--", lw=0.8)
        ax.set_ylabel(lab)
    axs[0].legend(fontsize=8)
    axs[0].set_title(f"Over time (dashed = table removed): {label}", fontsize=9)
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "timeseries.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- reaction test
    m, f = float(np.median(args.masses)), float(np.median(args.frictions))
    pull_step = removal + (env.unwrapped.max_episode_steps - removal) // 3
    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    deltas = {}
    for name, pull in [("no pull", 0.0), (f"+{args.pull:g} N pull", args.pull)]:
        s, log = run_episode(env, policy, m, f, seed=1000, pull_newton=pull, pull_step=pull_step)
        t = np.arange(len(log["grip"])) / 100.0
        axs[0].plot(t, log["grip"], label=f"{name} ({'held' if s['success'] else 'dropped'})")
        axs[1].plot(t, log["flex_cmd"])
        axs[2].plot(t, 100 * np.array(log["disp"]))
        before = np.mean(log["flex_cmd"][pull_step - 20 : pull_step])
        after = log["flex_cmd"][pull_step + 10 : pull_step + 60]
        deltas[name] = (np.mean(after) - before) if len(after) else np.nan
    for ax, lab in zip(axs, ["grip force (N)", "mean finger closing\ncommand (rad)", "slip (cm)"]):
        ax.axvline(removal / 100, c="k", ls="--", lw=0.8)
        ax.axvline(pull_step / 100, c="tab:red", ls=":", lw=1)
        ax.set_ylabel(lab)
    axs[0].legend(fontsize=8)
    axs[0].set_title(f"Reaction test, {1000 * m:g} g μ={f:g} (red = pull starts): {label}", fontsize=9)
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "reaction.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- text summary
    print(f"\n== {label} ==   object={args.object}, {args.episodes} episodes per cell")
    print("success by mass (rows) x friction (cols):")
    print("          " + "".join(f"{f:>7g}" for f in args.frictions))
    for i, m in enumerate(args.masses):
        print(f"  {1000 * m:5g} g " + "".join(f"{succ[i, j]:>7.0%}" for j in range(F)))
    print(f"overall success: {np.mean(ok):.0%}")
    print(f"adaptation (decision): closing-command slope vs required force = {slope_cmd:+.3f} rad/N "
          f"(0 = same command for every object; >0 = closes harder when more is needed)")
    print(f"outcome: grip force while holding {np.nanmean(force):.1f} N on average vs {np.mean(req):.2f} N "
          f"required (slope {slope_force:+.2f}; partly physics, see plot b)")
    react = deltas[f"+{args.pull:g} N pull"] - deltas["no pull"]
    print(f"reaction to +{args.pull:g} N pull: closing command changed by {react:+.3f} rad "
          f"(~0 = no reaction; >0 = grips harder)")
    print(f"plots + grid.csv in {out}/")


if __name__ == "__main__":
    main()
