"""How much grip does a grasp REALLY need? (measured, instead of the m*g/mu formula)

For each mass x friction cell: run the policy normally (grasp, table removal, 1 s of holding),
then weaken all finger motors in steps ("staircase") while the policy keeps running:

    motor cap  100% -> 90% -> ... -> 10% -> 5% -> 0%   of the config cap, 0.5 s per step

The object counts as slipping once it has moved more than --slip-mm since the staircase
started (or fell). The last step before that gives:

    min motor cap  (N*m per finger motor): the weakest motors that still hold. This is the
                   number that maps to the real hand (Feetech current limit).
    min grip force (N): total hand->object contact force at that step (same "grip_force"
                   as in the reward and in force_vs_required.png).

    python min_grip.py --scripted configs/residual_cube_grip.yaml   # -> runs/scripted_<config>/diagnose/
    python min_grip.py runs/<run>                                   # -> runs/<run>/diagnose/

Writes min_grip.csv, min_grip.png (builds on force_vs_required.png: the policy's actual grip
next to the measured minimum and the formula), and min_grip_staircase.png (what one staircase
looks like). If diagnose.py was already run for the same target, its grid.csv is used for the
"actual grip" points, so both plots show the same episodes' numbers.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import load_config, make_env  # noqa: E402
from diagnose import Policy  # noqa: E402

LEVELS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]  # fraction of the config cap


def staircase(env, policy, mass, friction, seed, settle_steps, level_steps, slip_m):
    """One episode. Returns a summary dict and a per-level log."""
    core = env.unwrapped
    cap0 = core.model.actuator_forcerange.copy()
    fingers = [i for i, n in enumerate(core.actuated_joint_names(strip_side=True)) if not n.startswith("wrist")]
    core.object_mass, core.object_friction = mass, friction
    obs, _ = env.reset(seed=seed)
    n_act = env.action_space.shape[0]
    summary = {"mass": mass, "friction": friction, "seed": seed,
               "required_formula": mass * 9.81 / friction, "cap_config": float(cap0[fingers[0], 1]),
               "grip_full_cap": np.nan, "min_cap_frac": np.nan, "min_cap_Nm": np.nan, "min_grip": np.nan,
               "held_at_full_cap": 0}
    levels = []
    try:
        # 1) normal episode up to table removal + settle (the policy sees exactly what it trained on)
        info, terminated = {}, False
        for _ in range(core.table_removal_step + settle_steps):
            obs, _, terminated, _, info = env.step(policy(obs, n_act))
            if terminated:
                break
        if terminated or info["displacement"] > core.hold_tolerance:
            return summary, levels  # cannot hold it even with full motors
        summary["held_at_full_cap"] = 1
        d0 = info["displacement"]

        # 2) staircase: weaker and weaker finger motors (the wrist keeps its cap)
        for frac in LEVELS:
            core.model.actuator_forcerange[fingers] = cap0[fingers] * frac
            grip, effort = [], []
            for _ in range(level_steps):
                obs, _, terminated, _, info = env.step(policy(obs, n_act))
                grip.append(info["reward_terms"]["grip_force"])
                effort.append(float(np.mean(np.abs(core.data.actuator_force[fingers]))))
                if terminated:
                    break
            slip = info["displacement"] - d0
            levels.append({"frac": frac, "grip": float(np.mean(grip[-20:])),  # last 0.2 s of the step
                           "effort": float(np.mean(effort[-20:])), "slip_mm": 1000 * slip})
            if terminated or slip > slip_m:
                break
    finally:
        core.model.actuator_forcerange[:] = cap0  # never leak a weakened model into the next episode

    summary["grip_full_cap"] = levels[0]["grip"]
    held = [lv for lv in levels if lv["slip_mm"] <= 1000 * slip_m]
    if held:
        last = held[-1]
        summary["min_cap_frac"] = last["frac"]
        summary["min_cap_Nm"] = last["frac"] * summary["cap_config"]
        summary["min_grip"] = last["grip"]
    return summary, levels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("target", nargs="?", help="run folder (runs/...)")
    p.add_argument("--scripted", metavar="CONFIG", help="measure the scripted grasp under this config")
    p.add_argument("--which", default="best_model")
    p.add_argument("--object", default="cube_4cm")
    p.add_argument("--masses", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.15])
    p.add_argument("--frictions", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.2])
    p.add_argument("--episodes", type=int, default=3, help="episodes per cell")
    p.add_argument("--jitter", type=float, default=0.003, help="start xy jitter (m), same as diagnose.py")
    p.add_argument("--settle", type=int, default=100, help="steps of normal holding before the staircase")
    p.add_argument("--level-steps", type=int, default=50, help="steps per staircase level (0.01 s each)")
    p.add_argument("--slip-mm", type=float, default=5.0, help="movement that counts as slipping")
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
    cfg["env"]["object_xy_jitter"] = args.jitter
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg)
    policy = Policy(cfg, run_dir, args.which)
    slip_m = args.slip_mm / 1000

    # ---------------------------------------------------------------- measure
    rows, example = [], None
    for m in args.masses:
        for f in args.frictions:
            for k in range(args.episodes):
                s, levels = staircase(env, policy, m, f, 1000 + k, args.settle, args.level_steps, slip_m)
                rows.append(s)
                # example plot: the middle cell's first episode (else any episode that was held)
                middle = m == args.masses[len(args.masses) // 2] and f == args.frictions[len(args.frictions) // 2]
                if levels and (example is None or (middle and k == 0)):
                    example = (s, levels)
    with open(out / "min_grip.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    req = np.array([r["required_formula"] for r in rows])
    min_grip = np.array([r["min_grip"] for r in rows])
    min_cap = np.array([r["min_cap_Nm"] for r in rows])
    held = ~np.isnan(min_grip)  # held with full motors AND for at least the first staircase step
    cap_cfg = rows[0]["cap_config"]

    # Actual grip while holding: from diagnose.py's grid.csv if present (same seeds/jitter), else
    # from the first staircase level (= full motors, i.e. normal behaviour).
    actual = np.array([r["grip_full_cap"] for r in rows])
    actual_src = "first staircase level"
    grid_csv = out / "grid.csv"
    if grid_csv.exists():
        with open(grid_csv) as fh:
            g = {(float(r["mass"]), float(r["friction"]), int(r["seed"])): float(r["grip_hold_mean"])
                 for r in csv.DictReader(fh)}
        if all((r["mass"], r["friction"], r["seed"]) in g for r in rows):
            actual = np.array([g[(r["mass"], r["friction"], r["seed"])] for r in rows])
            actual_src = "diagnose grid.csv"

    # ---------------------------------------------------------------- min_grip.png
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axs[0]
    x = np.linspace(0, req.max() * 1.1, 10)
    ax.plot(x, x, "k--", lw=1, label="formula m·g/μ")
    ax.plot(x, 2 * x, ":", c="gray", lw=1, label="2× formula (grip penalty starts above)")
    for i in np.nonzero(held)[0]:  # line = margin the policy keeps above what it needs
        ax.plot([req[i], req[i]], [min_grip[i], actual[i]], c="lightgray", lw=1, zorder=0)
    ax.scatter(req[held], actual[held], c="tab:blue", s=22, label=f"actual grip while holding ({actual_src})")
    ax.scatter(req[held], min_grip[held], c="tab:green", marker="v", s=30, label="measured minimum (just before slip)")
    if (~held).any():
        ax.scatter(req[~held], np.zeros((~held).sum()), c="tab:red", marker="x", s=30,
                   label="slips even at full motors")
    ax.set_xlabel("required by formula m·g/μ (N)")
    ax.set_ylabel("total contact force hand→object (N)")
    ax.set_title("(a) grip force: actual vs measured minimum vs formula", fontsize=9)
    ax.legend(fontsize=7)

    ax = axs[1]
    ax.axhline(cap_cfg, c="k", ls="--", lw=1, label=f"config cap {cap_cfg:g} N·m")
    ax.scatter(req[held], min_cap[held], c="tab:green", marker="v", s=30, label="weakest cap that still holds")
    ax.set_xlabel("required by formula m·g/μ (N)")
    ax.set_ylabel("finger motor cap (N·m per motor)")
    ax.set_ylim(-0.02, cap_cfg * 1.1)
    ax.set_title("(b) motor strength actually needed", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle(f"{label}   (slip = moved > {args.slip_mm:g} mm)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "min_grip.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- one staircase, explained
    if example is not None:
        s, levels = example
        caps = [cap_cfg * lv["frac"] for lv in levels]
        fig, axs = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
        axs[0].plot(caps, [lv["grip"] for lv in levels], "o-", label="contact force (N)")
        axs[0].plot(caps, [100 * lv["effort"] for lv in levels], "s-", label="mean |finger motor torque| ×100 (N·m)")
        axs[1].plot(caps, [lv["slip_mm"] for lv in levels], "o-", c="tab:red")
        axs[1].axhline(args.slip_mm, c="k", ls=":", lw=1, label="slip threshold")
        for ax in axs:
            if not np.isnan(s["min_cap_Nm"]):
                ax.axvline(s["min_cap_Nm"], c="tab:green", lw=1)
        axs[0].legend(fontsize=8)
        axs[1].legend(fontsize=8)
        axs[0].set_ylabel("force / torque")
        axs[1].set_ylabel("moved since start (mm)")
        axs[1].set_xlabel("finger motor cap (N·m)  ← motors get weaker")
        axs[1].invert_xaxis()
        axs[0].set_title(f"Staircase, {1000 * s['mass']:g} g μ={s['friction']:g} (green = weakest that holds): {label}",
                         fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "min_grip_staircase.png", dpi=120)
        plt.close(fig)

    # ---------------------------------------------------------------- text summary
    print(f"\n== {label} ==   object={args.object}, {args.episodes} episodes per cell, slip > {args.slip_mm:g} mm")
    print(f"{'mass':>6} {'mu':>5} {'formula':>8} {'actual':>7} {'min':>6} {'min/formula':>11} {'min cap':>8}")
    for m in args.masses:
        for f in args.frictions:
            sel = [i for i, r in enumerate(rows) if r["mass"] == m and r["friction"] == f]
            ok = [i for i in sel if held[i]]
            if not ok:
                print(f"{1000 * m:5g}g {f:5g} {req[sel[0]]:7.2f}N   slips/drops even at full motors")
                continue
            mg, ac, mc = np.nanmean(min_grip[ok]), np.nanmean(actual[ok]), np.nanmean(min_cap[ok])
            print(f"{1000 * m:5g}g {f:5g} {req[sel[0]]:7.2f}N {ac:6.1f}N {mg:5.1f}N {mg / req[sel[0]]:10.1f}× "
                  f"{mc:7.3f}  ({len(ok)}/{len(sel)} held)")
    if held.any():
        print(f"\nmeasured minimum grip: {np.nanmean(min_grip[held]):.1f} N on average "
              f"(formula: {np.mean(req[held]):.2f} N; actual: {np.nanmean(actual[held]):.1f} N)")
        print(f"weakest motor cap that holds: {np.nanmean(min_cap[held]):.3f} N·m on average "
              f"(config: {cap_cfg:g}; max over cells: {np.nanmax(min_cap[held]):.3f})")
    print(f"plots + min_grip.csv in {out}/")


if __name__ == "__main__":
    main()
