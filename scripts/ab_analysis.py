#!/usr/bin/env python3
"""Analyse an A/B test of the intervention — or simulate one to validate the estimator.

Real usage::

    python scripts/ab_analysis.py --input ab_test.csv

``ab_test.csv`` needs ``delay_days`` and ``treated`` (0/1), optionally
``shipping_mode``.  The script reports the overall reduction, a sample-ratio
check, per-mode effects with confidence intervals, and whether each segment's
*lower bound* clears the break-even effect from the cost model.

No real experiment exists for DataCo, so the default mode simulates one from
the observed per-mode delay distribution with known effects.  That validates
coverage and the rollout rule; it does not replace running the test.

Outputs: ``results/ab_analysis.json``, ``results/ab_by_mode.csv``,
``images/ab_analysis.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiment, policy, scoring  # noqa: E402

DEFAULT_EFFECTS = {
    "Same Day": 0.05,
    "First Class": 1.0,
    "Second Class": 2.0,
    "Standard Class": 2.0,
}


def mode_break_even(bundle: dict, penalty: float, express_cost: float) -> dict:
    return {
        mode: policy.break_even_reduction(
            np.atleast_2d(pmf), penalty, express_cost
        )
        for mode, pmf in bundle["mode_pmf"].items()
    }


def simulate_from_bundle(
    bundle: dict, effects: dict, n_per_arm: int, seed: int, noise: float
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    atoms = np.asarray(bundle["atoms"], dtype=float)
    pools = {
        mode: rng.choice(atoms, size=5000, p=np.asarray(pmf) / np.asarray(pmf).sum())
        for mode, pmf in bundle["mode_pmf"].items()
    }
    return experiment.simulate_ab(pools, effects, n_per_arm=n_per_arm, seed=seed, noise=noise)


def summarise(
    frame: pd.DataFrame,
    break_even: dict,
    penalty: float,
    express_cost: float,
    simulated: bool,
) -> tuple[dict, pd.DataFrame]:
    overall = experiment.welch_reduction(
        frame.loc[frame["treated"] == 0, "delay_days"],
        frame.loc[frame["treated"] == 1, "delay_days"],
    ).as_dict()
    srm = experiment.srm_test(
        int((frame["treated"] == 0).sum()), int((frame["treated"] == 1).sum())
    )

    if "shipping_mode" in frame.columns:
        by_mode = experiment.analyze_by_group(frame)
        by_mode["break_even"] = by_mode["shipping_mode"].map(break_even)
    else:
        by_mode = pd.DataFrame([{**overall, "shipping_mode": "all",
                                 "break_even": break_even.get("all", float("nan"))}])

    srm_ok = bool(srm.get("ok", True))
    decisions = []
    for low, high, be, mode in zip(  # noqa: B905
        by_mode["ci_low"], by_mode["ci_high"], by_mode["break_even"], by_mode["shipping_mode"]
    ):
        if not srm_ok:
            decisions.append("hold (SRM anomaly)")
        elif mode == "all" and not np.isfinite(be):
            decisions.append("not evaluated (no break-even)")
        else:
            decisions.append(experiment.rollout_decision(low, high, be))
    by_mode["decision"] = decisions
    if not srm_ok:
        srm["note"] = "SRM 检验未通过：分流比例异常，任何效应估计都不可信，已阻断上线建议"
    # 每个运输方式必须用**本组**的对照延迟：pooled 对照会混进 Same Day 这种
    # 本来就不晚的订单，把净收益压成 0（复审 S2 的最小反例）。
    control_all = frame.loc[frame["treated"] == 0, "delay_days"].to_numpy(dtype=float)
    ctrl_by_mode: dict = {}
    if "shipping_mode" in frame.columns:
        ctrl_by_mode = {
            mode: chunk.loc[chunk["treated"] == 0, "delay_days"].to_numpy(dtype=float)
            for mode, chunk in frame.groupby("shipping_mode")
        }
    net_values, upper_bounds = [], []
    for _, row in by_mode.iterrows():
        ctrl = ctrl_by_mode.get(row["shipping_mode"]) if ctrl_by_mode else control_all
        if ctrl is None or getattr(ctrl, "size", 0) == 0:
            ctrl = control_all
        net_values.append(experiment.net_value(row["reduction"], penalty, express_cost, ctrl))
        upper_bounds.append(experiment.net_value(row["reduction"], penalty, express_cost))
    by_mode["net_value_per_order"] = net_values
    by_mode["net_value_upper_bound"] = upper_bounds
    if simulated and "true_effect" in frame.columns:
        truth = frame.groupby("shipping_mode")["true_effect"].first()
        by_mode["true_effect"] = by_mode["shipping_mode"].map(truth)

    summary = {
        "mode": "simulated" if simulated else "observed",
        "overall": overall,
        "srm": srm,
        "by_mode": by_mode.to_dict(orient="records"),
        "params": {
            "penalty_per_day": penalty,
            "express_cost": express_cost,
            "n_control": int((frame["treated"] == 0).sum()),
            "n_treatment": int((frame["treated"] == 1).sum()),
        },
    }
    return summary, by_mode


def plot(by_mode: pd.DataFrame, images: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    view = by_mode.sort_values("break_even", na_position="first")
    y = np.arange(len(view))
    colors = {
        "roll out": "#1f6f78",
        "inconclusive": "#d99a2b",
        "do not roll out": "#9b3b3b",
        "never pays": "#7c8a99",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    ax = axes[0]
    for i, (_, row) in enumerate(view.iterrows()):
        ax.errorbar(
            row["reduction"],
            i,
            xerr=[[row["reduction"] - row["ci_low"]], [row["ci_high"] - row["reduction"]]],
            fmt="o",
            color=colors.get(row["decision"], "#333"),
            capsize=3,
        )
        if np.isfinite(row["break_even"]):
            ax.plot(row["break_even"], i, marker="|", color="#9b3b3b", markersize=14)
    ax.axvline(0, color="#c9ced6", linewidth=1)
    ax.set_yticks(y, view["shipping_mode"])
    ax.set_xlabel("delay reduction (days), 95% CI")
    ax.set_title("Effect vs break-even (red tick)")
    ax.legend(
        handles=[plt.Line2D([], [], marker="|", color="#9b3b3b", linestyle="", markersize=12,
                            label="break-even")],
        fontsize=8,
        loc="lower right",
    )

    ax2 = axes[1]
    values = view["net_value_per_order"]
    ax2.barh(y, values, color=[colors.get(d, "#333") for d in view["decision"]])
    ax2.axvline(0, color="#333", linewidth=1)
    ax2.set_yticks(y, view["shipping_mode"])
    ax2.set_xlabel("net value per treated order (USD)")
    ax2.set_title("Value at the point estimate")
    fig.tight_layout()
    fig.savefig(images / "ab_analysis.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=None, help="CSV with delay_days and treated (0/1)")
    ap.add_argument("--bundle", default=None, help="model bundle path")
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--images", default=str(ROOT / "images"))
    ap.add_argument("--penalty", type=float, default=None)
    ap.add_argument("--express-cost", type=float, default=100.0)
    ap.add_argument("--n-per-arm", type=int, default=1000, help="simulation only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effect", type=float, default=None, help="uniform simulated effect (days)")
    ap.add_argument("--noise", type=float, default=0.25, help="simulated per-order noise (days)")
    args = ap.parse_args()

    outdir, images = Path(args.outdir), Path(args.images)
    outdir.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    bundle = scoring.load_bundle(args.bundle)
    penalty = float(bundle["params"]["penalty"] if args.penalty is None else args.penalty)
    break_even = mode_break_even(bundle, penalty, args.express_cost)

    if args.input:
        frame = pd.read_csv(args.input, encoding="latin1")
        missing = {"delay_days", "treated"} - set(frame.columns)
        if missing:
            raise SystemExit(f"input is missing columns: {sorted(missing)}")
        frame["treated"] = frame["treated"].astype(int)
        simulated = False
    else:
        effects = (
            {mode: args.effect for mode in DEFAULT_EFFECTS}
            if args.effect is not None
            else DEFAULT_EFFECTS
        )
        frame = simulate_from_bundle(bundle, effects, args.n_per_arm, args.seed, args.noise)
        simulated = True

    summary, by_mode = summarise(frame, break_even, penalty, args.express_cost, simulated)
    by_mode.to_csv(outdir / "ab_by_mode.csv", index=False)
    (outdir / "ab_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    overall = summary["overall"]
    print(f"A/B analysis ({'simulated' if simulated else 'observed'}), "
          f"n = {summary['params']['n_control']:,} control / {summary['params']['n_treatment']:,} treated")
    print(f"  overall reduction {overall['reduction']:+.2f} days "
          f"[{overall['ci_low']:+.2f}, {overall['ci_high']:+.2f}], p={overall['p_value']:.2g}")
    print(f"  SRM p={summary['srm']['p_value']:.3f} "
          f"({'ok' if summary['srm']['ok'] else 'BROKEN randomisation'})")
    for _, row in by_mode.iterrows():
        break_even_text = (
            f"{row['break_even']:.2f}d" if np.isfinite(row["break_even"]) else "never"
        )
        print(f"  {row['shipping_mode']:<16s} {row['reduction']:+.2f}d "
              f"[{row['ci_low']:+.2f}, {row['ci_high']:+.2f}] "
              f"break-even {break_even_text:<6s} -> {row['decision']} "
              f"(${row['net_value_per_order']:+,.0f}/order)")

    try:
        plot(by_mode, images)
        print(f"wrote {images / 'ab_analysis.png'}")
    except Exception as exc:  # pragma: no cover - optional chart
        print(f"chart skipped: {exc}")
    print(f"wrote {outdir / 'ab_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
