#!/usr/bin/env python3
"""Heterogeneous effects: who benefits most from express?

The current policy assumes express removes two days for every order.  This
script simulates an experiment with a **feature-dependent** effect, fits a
T-learner, and compares three ways to pick the orders to treat:

* ``cate``     — estimated individual effect (uplift);
* ``p_late``   — predicted probability of being late (the classic risk score);
* ``pmf_gain`` — expected benefit under the current constant-effect model.

Evaluation uses the *true* simulated effect, so the comparison measures the
ranking rule, not the outcome model.  Qini/AUUC show the ranking quality; the
cost curve shows what it is worth in dollars.

Outputs: ``results/uplift.json``, ``results/uplift_targeting.csv``,
``images/uplift.png``.
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

from src import pipeline, policy, scoring, uplift  # noqa: E402


def true_effect(frame: pd.DataFrame, high_value_threshold: float) -> np.ndarray:
    """Simulated, feature-dependent reduction in delay days.

    High-value orders and slower shipping modes benefit more; Same Day orders
    barely benefit at all.  This is an assumption for the demo, not a claim
    about DataCo.
    """
    mode = frame["Shipping Mode"].to_numpy()
    value = frame["Order Item Total"].to_numpy(dtype=float)
    discount = frame["Order Item Discount Rate"].to_numpy(dtype=float)
    high_value = value > high_value_threshold

    effect = np.full(len(frame), 0.3)
    effect += 0.8 * (mode == "Second Class")
    effect += 0.7 * (mode == "Standard Class")
    effect -= 0.2 * (mode == "Same Day")
    effect += 1.2 * high_value
    effect += 0.3 * (discount > 0.1)
    return np.clip(effect, 0.02, None)


def simulate(base_delay: np.ndarray, effect: np.ndarray, seed: int, noise: float):
    rng = np.random.default_rng(seed)
    treated = rng.integers(0, 2, size=len(base_delay))
    outcome = base_delay - treated * effect + rng.normal(0, noise, size=len(base_delay))
    return treated, outcome


def pmf_gain(bundle: dict, frame: pd.DataFrame, penalty: float, action_cost: float) -> np.ndarray:
    pmf = policy.pmf_matrix(
        frame["Shipping Mode"].to_numpy(), bundle["mode_pmf"], bundle["mode_pmf_default"]
    )
    base = policy.expected_excess_penalty_pmf(pmf, reduction=0.0)
    reduced = policy.expected_excess_penalty_pmf(pmf, reduction=2.0)
    return penalty * (base - reduced) - action_cost


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None)
    ap.add_argument("--bundle", default=None, help="model bundle path")
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--images", default=str(ROOT / "images"))
    ap.add_argument("--penalty", type=float, default=None)
    ap.add_argument("--action-cost", type=float, default=100.0)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    outdir, images = Path(args.outdir), Path(args.images)
    outdir.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    bundle = scoring.load_bundle(args.bundle)
    penalty = float(bundle["params"]["penalty"] if args.penalty is None else args.penalty)

    df = pipeline.load_orders(str(pipeline.find_data(args.data)))
    features, target = pipeline.build_features(df)
    split = pipeline.time_split(features, target)

    train = split.train_index
    test = split.test_index
    X_train = features.iloc[train].to_numpy()
    X_test = features.iloc[test].to_numpy()
    df_train = df.iloc[train]
    df_test = df.iloc[test]

    high_value_threshold = float(np.quantile(df["Order Item Total"], 0.75))
    effect_train = true_effect(df_train, high_value_threshold)
    effect_test = true_effect(df_test, high_value_threshold)

    # Unobserved heterogeneity: part of the effect is not in the features, so
    # no model can rank it perfectly.  Without this, AUUC is artificially ~1.
    effect_rng = np.random.default_rng(args.seed + 1)
    effect_train = np.clip(effect_train + effect_rng.normal(0, 0.35, len(effect_train)), 0.02, None)
    effect_test = np.clip(effect_test + effect_rng.normal(0, 0.35, len(effect_test)), 0.02, None)
    treated_train, outcome_train = simulate(target[train], effect_train, args.seed, args.noise)

    model = uplift.fit_t_learner(X_train, treated_train, outcome_train)
    cate = model.predict_cate(X_test)

    p_late = bundle["binary"].predict_proba(split.X_test)[:, 1]
    gain = pmf_gain(bundle, df_test, penalty, args.action_cost)
    rng = np.random.default_rng(args.seed)
    scores = {
        "cate": cate,
        "p_late": p_late,
        "pmf_gain": gain,
        "random": rng.uniform(size=len(cate)),
        "oracle": effect_test,
    }

    curve = uplift.qini_curve(cate, effect_test)
    qini_auc = uplift.auuc(curve)

    targeting = []
    for name, score in scores.items():
        frame = uplift.targeting_curve(score, effect_test, penalty, args.action_cost, base_delay=target[test].to_numpy())
        frame["score"] = name
        targeting.append(frame)
    targeting_df = pd.concat(targeting, ignore_index=True)
    targeting_df.to_csv(outdir / "uplift_targeting.csv", index=False)

    summary = {
        "true_ate": float(effect_test.mean()),
        "estimated_ate": float(cate.mean()),
        "auuc": float(qini_auc),
        "params": {
            "penalty_per_day": penalty,
            "action_cost": args.action_cost,
            "noise": args.noise,
            "seed": args.seed,
        },
        "net_saving": {
            name: {
                f"{fraction:.0%}": float(
                    group.loc[np.isclose(group["fraction"], fraction), "net_saving"].iloc[0]
                )
                for fraction in (0.10, 0.20, 0.30)
            }
            for name, group in targeting_df.groupby("score")
        },
    }
    (outdir / "uplift.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"true ATE {summary['true_ate']:.2f} days, estimated ATE {summary['estimated_ate']:.2f} days")
    print(f"Qini AUUC (CATE) {qini_auc:.3f}")
    for fraction in (0.10, 0.20, 0.30):
        print(f"  net saving at {fraction:.0%} treated:")
        for name in ("cate", "p_late", "pmf_gain", "random", "oracle"):
            value = summary["net_saving"][name][f"{fraction:.0%}"]
            print(f"    {name:<9s} ${value:>12,.0f}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
        axes[0].plot(curve["fraction"], curve["qini"], color="#1f6f78", label="CATE ranking")
        axes[0].plot(curve["fraction"], curve["perfect"], color="#9b3b3b", linestyle="--",
                     label="perfect")
        axes[0].axhline(0, color="#c9ced6", linewidth=1, label="random")
        axes[0].set_xlabel("fraction of orders treated")
        axes[0].set_ylabel("cumulative uplift (days)")
        axes[0].set_title(f"Qini curve — AUUC {qini_auc:.2f}")
        axes[0].legend(fontsize=8)

        colors = {"cate": "#1f6f78", "p_late": "#d99a2b", "pmf_gain": "#9b3b3b",
                  "random": "#c9ced6", "oracle": "#2f5d8c"}
        for name, group in targeting_df.groupby("score"):
            axes[1].plot(group["fraction"] * 100, group["net_saving"] / 1e3,
                          marker="o", markersize=3, label=name, color=colors.get(name))
        axes[1].axhline(0, color="#333", linewidth=1)
        axes[1].set_xlabel("orders treated (%)")
        axes[1].set_ylabel("net saving (USD k)")
        axes[1].set_title("Value of targeting (true effects)")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(images / "uplift.png", dpi=150)
        plt.close(fig)
        print(f"wrote {images / 'uplift.png'}")
    except Exception as exc:  # pragma: no cover - optional chart
        print(f"chart skipped: {exc}")

    print(f"wrote {outdir / 'uplift.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
