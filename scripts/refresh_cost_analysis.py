#!/usr/bin/env python3
"""Reproduce every number in the README, from raw data to cost table.

What this script does
---------------------
1. Loads DataCo, builds order-time-only features, splits chronologically.
2. Fits three quantile regressors (P10/P50/P90) — the original approach.
3. Fits a binary late/on-time classifier — the classic baseline.
4. Fits a *distribution* model (7-class classifier over the delay atoms) —
   the fix for the discrete-target problem.
5. Diagnoses the information ceiling: how much AUC is left beyond shipping mode.
6. Scores ten policies under one cost function and bootstraps the deltas.
7. Writes ``results/``, ``images/`` and the model bundle used by the app.

Usage
-----
    python scripts/refresh_cost_analysis.py                 # full run (~3 min first time)
    python scripts/refresh_cost_analysis.py --fast          # tiny models, for CI
    python scripts/refresh_cost_analysis.py --penalty 100 --no-cache

Outputs
-------
    results/policy_comparison.csv
    results/bootstrap.csv
    results/sensitivity.csv
    results/capacity_curve.csv
    results/budget_curve.csv
    results/robustness.csv
    results/robustness_summary.csv
    results/cost_model_check.csv
    results/mode_table.csv
    results/diagnostics.json
    results/summary.md
    models/model_bundle.joblib      (consumed by app/app.py)
    images/policy_costs.png
    images/sensitivity.png
    images/signal_ceiling.png
    images/distribution_example.png
    images/capacity.png
    images/budget.png
    images/decision_map.png
    images/robustness.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline, policy  # noqa: E402

CACHE_VERSION = "pmf-v2"
CACHE = ROOT / "models" / "quantile_cache.joblib"
BUNDLE = ROOT / "models" / "model_bundle.joblib"


# --------------------------------------------------------------------------- data


def fingerprint(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{CACHE_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def code_fingerprint() -> str:
    """Hash of the modules that define the features and the cost function.

    Without this, editing feature engineering silently reuses a model trained
    on the old features — the exact foot-gun the v1 cache had.
    """
    digest = hashlib.sha1()
    for rel in ("src/pipeline.py", "src/policy.py"):
        digest.update((ROOT / rel).read_bytes())
    return digest.hexdigest()[:12]


# --------------------------------------------------------------------------- models


def fit_or_load(split, args, data_fp: str):
    """Train (or load) quantile, binary and pmf models.

    The cache key covers the data file *and* the model hyper-parameters: a
    stale cache was a real foot-gun in v1, where editing features silently
    reused an old model.
    """
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
    )

    key = {
        "cache_version": CACHE_VERSION,
        "data_fp": data_fp,
        "n_estimators": args.n_estimators,
        "pmf_iter": args.pmf_iter,
        "features": len(split.feature_names),
        "code": code_fingerprint(),
    }
    key_str = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]

    if CACHE.exists() and not args.no_cache:
        payload = joblib.load(CACHE)
        if payload.get("cache_key") == key_str:
            print(f"[cache] hit {CACHE.relative_to(ROOT)} ({key_str})")
            return payload
        print("[cache] stale cache ignored (data or params changed)")

    models = {}
    for q in (0.1, 0.5, 0.9):
        t0 = time.time()
        model = GradientBoostingRegressor(
            n_estimators=args.n_estimators,
            max_depth=8,
            learning_rate=0.05,
            loss="quantile",
            alpha=q,
            random_state=42,
        )
        model.fit(split.X_train, split.y_train)
        models[q] = model
        print(f"  quantile q={q}: fitted in {time.time() - t0:.0f}s")

    from sklearn.metrics import roc_auc_score

    try:
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=40 if args.fast else 300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
        clf.fit(split.X_train, (split.y_train > 0).astype(int))
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"  xgboost unavailable ({exc}); using GradientBoostingClassifier")
        clf = GradientBoostingClassifier(
            n_estimators=args.n_estimators, max_depth=6, random_state=42
        )
        clf.fit(split.X_train, (split.y_train > 0).astype(int))

    auc = roc_auc_score((split.y_test > 0).astype(int), clf.predict_proba(split.X_test)[:, 1])
    print(f"  binary AUC (time split) = {auc:.4f}")

    # The delay is an integer between -2 and 4, so a 7-class classifier is the
    # natural distribution model.  HistGradientBoosting keeps the fit ~10s.
    t0 = time.time()
    pmf_model = HistGradientBoostingClassifier(
        max_iter=args.pmf_iter,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42,
    )
    pmf_model.fit(split.X_train, split.y_train.astype(int))
    print(f"  pmf (7-class) fitted in {time.time() - t0:.0f}s")

    payload = {
        "cache_key": key_str,
        "quantile": models,
        "binary": clf,
        "pmf_model": pmf_model,
        "binary_auc": auc,
        "features": split.feature_names,
    }
    if not args.no_cache:
        CACHE.parent.mkdir(exist_ok=True)
        joblib.dump(payload, CACHE)
    return payload


def pmf_from_classifier(model, X_test, atoms=policy.ATOMS) -> np.ndarray:
    """Align a classifier's ``predict_proba`` columns onto the atom grid."""
    proba = model.predict_proba(X_test)
    out = np.zeros((len(X_test), len(atoms)))
    for j, cls in enumerate(model.classes_):
        hits = np.where(atoms == int(cls))[0]
        if hits.size:
            out[:, hits[0]] = proba[:, j]
    return out


# --------------------------------------------------------------------------- diagnostics


def chi2_test(frame: pd.DataFrame, feature: str, target: str = "delay_days") -> dict:
    """Chi-square independence test + Cramér's V (effect size)."""
    from scipy.stats import chi2_contingency

    table = pd.crosstab(frame[feature], frame[target])
    if min(table.shape) < 2:
        return {"feature": feature, "p_value": float("nan"), "cramers_v": float("nan"), "dof": 0}
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return {"feature": feature, "p_value": float(p), "cramers_v": v, "dof": int(dof)}


def signal_ceiling(df, split, payload, p_late, args) -> dict:
    """Quantify how much signal exists beyond ``Shipping Mode``.

    If the delay is uniform *inside* each shipping mode, then no feature
    engineering can help and the AUC of the full model must equal the AUC of a
    base-rate lookup.  This function measures exactly that.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    mode_train = df.loc[split.train_index, "Shipping Mode"].to_numpy()
    mode_test = df.loc[split.test_index, "Shipping Mode"].to_numpy()
    y_bin = (split.y_test > 0).astype(int)

    rates = pd.Series((split.y_train > 0).astype(float)).groupby(mode_train).mean()
    base_score = np.array([rates[m] for m in mode_test])
    base_auc = roc_auc_score(y_bin, base_score)
    model_auc = roc_auc_score(y_bin, p_late)

    within = {}
    for mode in np.unique(mode_train):
        tr = mode_train == mode
        te = mode_test == mode
        y_tr, y_te = (split.y_train[tr] > 0).astype(int), y_bin[te]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue  # First Class is always late, Same Day never is
        clf = HistGradientBoostingClassifier(
            max_iter=40 if args.fast else 100, max_depth=4, random_state=42
        )
        clf.fit(split.X_train[tr], y_tr)
        within[mode] = {
            "n_test": int(te.sum()),
            "late_rate": float(y_te.mean()),
            "auc": float(roc_auc_score(y_te, clf.predict_proba(split.X_test[te])[:, 1])),
        }

    std = df[df["Shipping Mode"] == "Standard Class"]
    tests = [
        chi2_test(std, col)
        for col in ("Market", "Category Name", "Customer Segment", "Type")
    ]
    tests.append(chi2_test(std.assign(order_month=std[pipeline.DATE_COL].dt.month), "order_month"))

    return {
        "mode_base_rate_auc": float(base_auc),
        "model_auc": float(model_auc),
        "auc_gap": float(model_auc - base_auc),
        "within_mode": within,
        "independence_tests": tests,
    }


# --------------------------------------------------------------------------- costing


def realised(idx, delay, penalty, actions=policy.ACTIONS) -> np.ndarray:
    return policy.realised_cost(idx, delay, penalty, actions)


def strategy_costs(delay, p, actions, *, draws, p_late, pos_pmf, neg_pmf, pmf) -> dict:
    """Cost of every strategy under an arbitrary (penalty, express-cost) pair."""
    n = len(delay)
    idx = {
        "none": np.full(n, policy.IDX_NONE),
        "all_express": np.full(n, policy.IDX_EXPRESS),
        "binary_cost_aware": policy.policy_cost_aware_binary(
            p_late, pos_pmf, neg_pmf, p, actions=actions
        ),
        "quantile_grid": policy.policy_expected_cost(draws, p, actions=actions),
        "pmf_exact": policy.policy_expected_cost_pmf(pmf, p, actions=actions),
    }
    return {k: float(realised(v, delay, p, actions).sum()) for k, v in idx.items()}


# --------------------------------------------------------------------------- report


def markdown_policy_table(comp: pd.DataFrame) -> str:
    view = comp[
        [
            "policy",
            "total_cost",
            "cost_per_order",
            "intervention_rate",
            "orders_notify",
            "orders_express",
            "recall_late",
            "precision",
        ]
    ].copy()
    view["total_cost"] = view["total_cost"].map(lambda v: f"${v:,.0f}")
    view["cost_per_order"] = view["cost_per_order"].map(lambda v: f"${v:,.2f}")
    for col in ("intervention_rate", "recall_late", "precision"):
        view[col] = view[col].map(lambda v: f"{v:.1%}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "|" + "|".join(["---"] * len(view.columns)) + "|"
    body = "\n".join("| " + " | ".join(map(str, row)) + " |" for row in view.values)
    return "\n".join([header, sep, body])


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Small markdown table writer (avoids the optional ``tabulate`` dependency)."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(map(str, cols)) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in df.itertuples(index=False):
        cells = [format(v, floatfmt) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to DataCoSupplyChainDataset.csv")
    ap.add_argument("--penalty", type=float, default=200.0, help="$ per late day")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--fast", action="store_true", help="small models for CI smoke tests")
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--images", default=str(ROOT / "images"))
    ap.add_argument("--bundle", default=str(BUNDLE), help="where to write the app model bundle")
    args = ap.parse_args()

    if args.fast:
        args.n_estimators, args.pmf_iter = 40, 30
    else:
        args.n_estimators, args.pmf_iter = 200, 120

    results_dir = Path(args.outdir)
    images_dir = Path(args.images)
    results_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    data_path = pipeline.find_data(args.data)
    print(f"== data ==\n{data_path}")
    df = pipeline.load_orders(str(data_path))
    facts = pipeline.describe(df)
    print(json.dumps(facts, indent=2, ensure_ascii=False))

    encoders = pipeline.fit_encoders(df)
    X, y = pipeline.build_features(df, encoders)
    split = pipeline.time_split(X, y)
    print(f"train={len(split.y_train):,} valid={len(split.y_valid):,} test={len(split.y_test):,}")
    print(f"features={len(split.feature_names)}")

    print("== models ==")
    payload = fit_or_load(split, args, fingerprint(data_path))

    preds = {q: m.predict(split.X_test) for q, m in payload["quantile"].items()}
    p10, p50, p90 = preds[0.1], preds[0.5], preds[0.9]
    draws = policy.quantile_grid(p10, p50, p90)
    delay = split.y_test
    pen = args.penalty
    n = len(delay)

    p_late = payload["binary"].predict_proba(split.X_test)[:, 1]
    p_late_valid = payload["binary"].predict_proba(split.X_valid)[:, 1]
    pmf_model = pmf_from_classifier(payload["pmf_model"], split.X_test)

    pos_pmf = policy.empirical_pmf(split.y_train[split.y_train > 0])
    neg_pmf = policy.empirical_pmf(split.y_train[split.y_train <= 0])
    mode_train = df.loc[split.train_index, "Shipping Mode"].to_numpy()
    mode_test = df.loc[split.test_index, "Shipping Mode"].to_numpy()
    mode_pmf_table, mode_pmf_default = policy.pmf_table(mode_train, split.y_train)
    pmf_lookup = policy.pmf_matrix(mode_test, mode_pmf_table, mode_pmf_default)

    # Cost-model assumption check: the flat $200/day is a choice, not a fact.
    # A value-proportional late fee is the other obvious model; see whether the
    # intervention story survives it.
    order_value = df.loc[split.test_index, "Order Item Total"].to_numpy(dtype=float)
    penalty_value = np.maximum(0.01 * order_value, 20.0)
    idx_value = policy.policy_expected_cost_pmf(pmf_lookup, penalty_value)
    value_mix = policy.action_mix(idx_value)
    flat_mix = policy.action_mix(policy.policy_expected_cost_pmf(pmf_lookup, pen))
    cost_model = pd.DataFrame(
        [
            {
                "cost_model": f"flat ${pen:.0f}/day",
                "median_penalty_per_day": float(pen),
                "intervention_rate": 1 - flat_mix["none"],
                "notify_rate": flat_mix["notify"],
                "express_rate": flat_mix["express"],
            },
            {
                "cost_model": "1% of order value/day (floor $20)",
                "median_penalty_per_day": float(np.median(penalty_value)),
                "intervention_rate": 1 - value_mix["none"],
                "notify_rate": value_mix["notify"],
                "express_rate": value_mix["express"],
            },
        ]
    )
    cost_model.to_csv(results_dir / "cost_model_check.csv", index=False)

    atoms = {int(v): float((delay <= v).mean()) for v in (-1, 0, 1, 2, 3)}
    print("achievable coverage ladder (test set):",
          " ".join(f"P(y<={k})={v:.1%}" for k, v in atoms.items()))
    print(f"achieved: P10={(delay <= p10).mean():.1%} "
          f"P50={(delay <= p50).mean():.1%} P90={(delay <= p90).mean():.1%}")
    print(f"MAE(P50) = {np.abs(delay - p50).mean():.3f} days")
    print("cost-model check:")
    for _, row in cost_model.iterrows():
        print(f"  {row['cost_model']:<38s} median fee ${row['median_penalty_per_day']:>6,.0f}/day "
              f"-> intervention {row['intervention_rate']:.1%}")

    # ------------------------------------------------------------------ diagnostics
    print("== diagnostics: how much signal is there? ==")
    ceiling = signal_ceiling(df, split, payload, p_late, args)
    print(f"  mode-base-rate AUC = {ceiling['mode_base_rate_auc']:.4f}")
    print(f"  model AUC           = {ceiling['model_auc']:.4f}  (gap {ceiling['auc_gap']:+.4f})")
    for mode, info in ceiling["within_mode"].items():
        print(f"  within {mode:<15s} AUC={info['auc']:.4f} (n={info['n_test']:,})")

    mode_table = (
        pd.DataFrame(
            {
                "shipping_mode": df.loc[split.train_index, "Shipping Mode"].to_numpy(),
                "delay_days": split.y_train,
            }
        )
        .groupby("shipping_mode")
        .agg(
            orders=("delay_days", "size"),
            late_rate=("delay_days", lambda s: float((s > 0).mean())),
            mean_delay=("delay_days", "mean"),
            min_delay=("delay_days", "min"),
            max_delay=("delay_days", "max"),
        )
        .reset_index()
    )
    mode_table["share"] = mode_table["orders"] / mode_table["orders"].sum()
    mode_table.to_csv(results_dir / "mode_table.csv", index=False)

    # ------------------------------------------------------------------ policies
    tau, val_cost = policy.tune_threshold(p_late_valid, split.y_valid, pen)
    print(f"  tuned binary threshold tau*={tau:.2f} (validation cost ${val_cost:,.0f})")

    strategies = {
        "none (do nothing)": policy.policy_none(n),
        "all express": policy.policy_all_express(n),
        "binary: p50 > 0 (legacy)": policy.policy_binary(p50=p50),
        f"binary: tuned tau={tau:.2f}": np.where(p_late > tau, policy.IDX_EXPRESS, policy.IDX_NONE),
        "binary: cost-aware": policy.policy_cost_aware_binary(p_late, pos_pmf, neg_pmf, pen),
        "quantile grid (3 points)": policy.policy_expected_cost(draws, pen),
        "pmf exact (lookup)": policy.policy_expected_cost_pmf(pmf_lookup, pen),
        "pmf 7-class model": policy.policy_expected_cost_pmf(pmf_model, pen),
        "budget $500k (pmf)": policy.policy_budget_pmf(pmf_lookup, pen, 500_000),
        "capacity 25% (best-first)": policy.policy_capacity_pmf(pmf_lookup, pen, 0.25),
        "oracle (hindsight)": policy.policy_oracle(delay, pen),
    }
    comp = policy.compare(strategies, delay, pen).sort_values("total_cost").reset_index(drop=True)
    comp.to_csv(results_dir / "policy_comparison.csv", index=False)
    print("\n" + markdown_policy_table(comp))

    # ------------------------------------------------------------------ bootstrap
    print("\n== paired bootstrap vs the best feasible policy ==")
    ref_name = "pmf exact (lookup)"
    ref_cost = realised(strategies[ref_name], delay, pen)
    rows = []
    for name, idx in strategies.items():
        if name == ref_name:
            continue
        stats = policy.bootstrap_diff(
            realised(idx, delay, pen), ref_cost, n_boot=500 if args.fast else 2000
        )
        rows.append(
            {
                "policy": name,
                "mean_diff_per_order": stats["mean_diff"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "significant": stats["significant"],
            }
        )
    boot = pd.DataFrame(rows).sort_values("mean_diff_per_order")
    boot.to_csv(results_dir / "bootstrap.csv", index=False)
    for _, r in boot.iterrows():
        star = "*" if r["significant"] else " "
        print(f"  {star} {r['policy']:<28s} {r['mean_diff_per_order']:+8.2f}/order "
              f"[{r['ci_low']:+7.2f}, {r['ci_high']:+7.2f}]")

    # ------------------------------------------------------------------ robustness
    print("\n== robustness: uncertainty in the cost parameters ==")
    scenario_sets = {
        "wide band (penalty 100-400, express 50-150)": ([100, 200, 300, 400], [50, 100, 150]),
        "narrow band (nominal +/-25%)": ([150, 200, 250], [75, 100, 125]),
    }
    idx_nominal = strategies["pmf exact (lookup)"]
    orders_ix = np.arange(n)
    rob_rows = []
    rob_summary = []
    for set_name, (penalties, express_costs) in scenario_sets.items():
        idx_robust = policy.policy_minimax_regret(pmf_lookup, penalties, express_costs)
        scenario_cost = policy.scenario_cost_matrix(pmf_lookup, penalties, express_costs)
        scenario_best = scenario_cost.min(axis=2)
        regret_nominal = scenario_cost[:, orders_ix, idx_nominal] - scenario_best
        regret_robust = scenario_cost[:, orders_ix, idx_robust] - scenario_best

        scenario_index = 0
        set_rows = []
        for p in penalties:
            for c in express_costs:
                actions = (policy.NONE, policy.NOTIFY, policy.Action("express", cost=c, reduction=2.0))
                cost_nom = float(realised(idx_nominal, delay, p, actions).sum())
                cost_rob = float(realised(idx_robust, delay, p, actions).sum())
                cost_opt = float(
                    realised(
                        policy.policy_expected_cost_pmf(pmf_lookup, p, actions=actions),
                        delay, p, actions,
                    ).sum()
                )
                set_rows.append(
                    {
                        "scenario_set": set_name,
                        "penalty_per_day": p,
                        "express_cost": c,
                        "cost_scenario_optimal": cost_opt,
                        "cost_nominal_policy": cost_nom,
                        "cost_robust_policy": cost_rob,
                        "regret_nominal_expected": float(regret_nominal[scenario_index].mean()),
                        "regret_robust_expected": float(regret_robust[scenario_index].mean()),
                    }
                )
                scenario_index += 1
        rob_rows.extend(set_rows)

        nominal_row = next(
            row for row in set_rows if row["penalty_per_day"] == 200 and row["express_cost"] == 100
        )
        mix = policy.action_mix(idx_robust)
        rob_summary.append(
            {
                "scenario_set": set_name,
                "worst_regret_nominal": float(regret_nominal.max(axis=0).mean()),
                "worst_regret_robust": float(regret_robust.max(axis=0).mean()),
                "mean_regret_nominal": float(regret_nominal.mean()),
                "mean_regret_robust": float(regret_robust.mean()),
                "premium_at_nominal": float(nominal_row["cost_robust_policy"] - nominal_row["cost_nominal_policy"]),
                "robust_none_rate": mix["none"],
                "robust_notify_rate": mix["notify"],
                "robust_express_rate": mix["express"],
            }
        )
    rob = pd.DataFrame(rob_rows)
    rob_summary_df = pd.DataFrame(rob_summary)
    rob.to_csv(results_dir / "robustness.csv", index=False)
    rob_summary_df.to_csv(results_dir / "robustness_summary.csv", index=False)
    for _, row in rob_summary_df.iterrows():
        print(f"  {row['scenario_set']}:")
        print(f"    worst-case regret ${row['worst_regret_nominal']:.2f} -> ${row['worst_regret_robust']:.2f} /order, "
              f"premium at nominal ${row['premium_at_nominal']:,.0f}, "
              f"mix none/notify/express {row['robust_none_rate']:.0%}/{row['robust_notify_rate']:.0%}/{row['robust_express_rate']:.0%}")

    # ------------------------------------------------------------------ sensitivity
    print("\n== sensitivity: when is the model worth it? ==")
    penalties = [100, 200, 300, 400]
    express_costs = [50, 75, 100, 125, 150]
    sens_rows = []
    for p in penalties:
        for c in express_costs:
            actions = (policy.NONE, policy.NOTIFY, policy.Action("express", cost=c, reduction=2.0))
            costs = strategy_costs(
                delay, p, actions,
                draws=draws, p_late=p_late, pos_pmf=pos_pmf, neg_pmf=neg_pmf, pmf=pmf_lookup,
            )
            best = min(costs, key=costs.get)
            idx_pmf = policy.policy_expected_cost_pmf(pmf_lookup, p, actions=actions)
            idx_q = policy.policy_expected_cost(draws, p, actions=actions)
            mix = policy.action_mix(idx_pmf, actions)
            sens_rows.append(
                {
                    "penalty_per_day": p,
                    "express_cost": c,
                    "cost_none": costs["none"],
                    "cost_all_express": costs["all_express"],
                    "cost_binary_cost_aware": costs["binary_cost_aware"],
                    "cost_quantile_grid": costs["quantile_grid"],
                    "cost_pmf_exact": costs["pmf_exact"],
                    "advantage_pmf_vs_quantile_pct": 1 - costs["pmf_exact"] / costs["quantile_grid"],
                    "advantage_pmf_vs_binary_pct": 1 - costs["pmf_exact"] / costs["binary_cost_aware"],
                    "best_policy": best,
                    "intervention_rate_pmf": 1 - mix["none"],
                    "notify_rate_pmf": mix["notify"],
                    "express_rate_pmf": mix["express"],
                    "intervention_rate_quantile": float((idx_q != policy.IDX_NONE).mean()),
                }
            )
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(results_dir / "sensitivity.csv", index=False)

    # ------------------------------------------------------------------ capacity
    capacities = [0.10, 0.25, 0.50, 0.75, 1.00]
    cap_rows = []
    for cap in capacities:
        idx_pmf = policy.policy_capacity_pmf(pmf_lookup, pen, cap)
        idx_q = policy.policy_capacity(draws, pen, cap)
        cap_rows.append(
            {
                "capacity": cap,
                "cost_pmf": float(realised(idx_pmf, delay, pen).sum()),
                "cost_quantile": float(realised(idx_q, delay, pen).sum()),
                "recall_pmf": float(((idx_pmf != policy.IDX_NONE) & (delay > 0)).sum() / (delay > 0).sum()),
                "recall_quantile": float(((idx_q != policy.IDX_NONE) & (delay > 0)).sum() / (delay > 0).sum()),
            }
        )
    cap = pd.DataFrame(cap_rows)
    cap.to_csv(results_dir / "capacity_curve.csv", index=False)

    # ------------------------------------------------------------------ budget
    budgets = [0, 100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_707_800]
    bc = policy.budget_curve(pmf_lookup, delay, pen, budgets)
    bc.to_csv(results_dir / "budget_curve.csv", index=False)

    # ------------------------------------------------------------------ figures
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. policy cost bars, coloured by family
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        sub = comp[~comp["policy"].str.startswith("oracle")]
        colors = []
        for name in sub["policy"]:
            if name.startswith("pmf"):
                colors.append("#1f6f78")
            elif name.startswith("binary"):
                colors.append("#d99a2b")
            elif name.startswith("quantile"):
                colors.append("#9b3b3b")
            elif "capacity" in name:
                colors.append("#7c8a99")
            else:
                colors.append("#c9ced6")
        ax.barh(sub["policy"], sub["total_cost"] / 1e3, color=colors)
        ax.set_xlabel("total cost on test set ($k)")
        ax.set_title(f"One cost function, {n:,} test orders, penalty ${pen:.0f}/day")
        for y_, v in enumerate(sub["total_cost"] / 1e3):
            ax.text(v + 2, y_, f"${v:,.0f}k", va="center", fontsize=8)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(images_dir / "policy_costs.png", dpi=150)
        plt.close(fig)

        # 2. signal ceiling
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        labels, values = ["shipping-mode\nbase rate", "full model\n(18 features)"], [
            ceiling["mode_base_rate_auc"], ceiling["model_auc"]
        ]
        for mode, info in ceiling["within_mode"].items():
            labels.append(f"within\n{mode}")
            values.append(info["auc"])
        bars = ax.bar(labels, values, color=["#1f6f78", "#2f5d8c", "#d99a2b", "#d99a2b"])
        ax.axhline(0.5, color="#9b3b3b", linestyle="--", linewidth=1, label="no signal (AUC 0.50)")
        ax.set_ylim(0.4, 0.85)
        ax.set_ylabel("ROC AUC")
        ax.set_title("Where the 0.78 ceiling comes from")
        ax.legend(loc="upper right", fontsize=8)
        for b, v in zip(bars, values):  # noqa: B905 (py3.9 compat)
            ax.text(b.get_x() + b.get_width() / 2, max(v, 0.505) + 0.012, f"{v:.3f}",
                    ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(images_dir / "signal_ceiling.png", dpi=150)
        plt.close(fig)

        # 3. sensitivity: cost vs express price at $200/day
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        view = sens[sens["penalty_per_day"] == 200]
        for col, label, color in [
            ("cost_binary_cost_aware", "binary + cost-aware rule", "#d99a2b"),
            ("cost_quantile_grid", "quantile grid (3 points)", "#9b3b3b"),
            ("cost_pmf_exact", "pmf (exact distribution)", "#1f6f78"),
            ("cost_all_express", "all express", "#c9ced6"),
        ]:
            ax.plot(view["express_cost"], view[col] / 1e3, marker="o", label=label, color=color)
        ax.set_xlabel("express cost ($/order)")
        ax.set_ylabel("total cost ($k)")
        ax.set_title("Cost vs express price, penalty $200/day")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(images_dir / "sensitivity.png", dpi=150)
        plt.close(fig)

        # 4. why three points are not a distribution
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
        example = int(np.argmax(mode_test == "Standard Class"))
        q_grid = policy._QGRID
        axes[0].bar(policy.ATOMS, pmf_lookup[example], color="#1f6f78", width=0.7)
        axes[0].set_title("true pmf: Standard Class delay (estimated on train)")
        axes[0].set_xlabel("delay (days)")
        axes[0].set_ylabel("probability")
        axes[0].set_ylim(0, 0.45)

        true_cdf = np.cumsum(pmf_lookup[example])
        axes[1].step(policy.ATOMS, true_cdf, where="post", color="#1f6f78",
                     label="true CDF (5 atoms)")
        grid = policy.quantile_grid(
            np.array([p10[example]]), np.array([p50[example]]), np.array([p90[example]])
        )[0]
        axes[1].plot(grid, q_grid, color="#9b3b3b", label="quantile-grid CDF")
        axes[1].set_title("same order, reconstructed from 3 quantiles")
        axes[1].set_xlabel("delay (days)")
        axes[1].set_ylim(0, 1)
        axes[1].legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(images_dir / "distribution_example.png", dpi=150)
        plt.close(fig)

        # 4b. robustness: regret across cost scenarios, two uncertainty bands
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
        for ax, (set_name, group) in zip(axes, rob.groupby("scenario_set", sort=False)):  # noqa: B905 (py3.9 compat)
            labels = [
                f"p{p}/c{c}"
                for p, c in zip(group["penalty_per_day"], group["express_cost"])  # noqa: B905 (py3.9 compat)
            ]
            x = np.arange(len(labels))
            ax.bar(x - 0.2, group["regret_nominal_expected"], width=0.4,
                   label="nominal policy", color="#9b3b3b")
            ax.bar(x + 0.2, group["regret_robust_expected"], width=0.4,
                   label="minimax-regret policy", color="#1f6f78")
            ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
            ax.set_title(set_name, fontsize=9)
            ax.set_ylabel("mean regret vs scenario optimum (USD/order)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(images_dir / "robustness.png", dpi=150)
        plt.close(fig)

        # 5. capacity curve: where ranking quality shows up
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
        axes[0].plot(cap["capacity"] * 100, cap["cost_pmf"] / 1e3, marker="o",
                     color="#1f6f78", label="pmf exact")
        axes[0].plot(cap["capacity"] * 100, cap["cost_quantile"] / 1e3, marker="o",
                     color="#9b3b3b", label="quantile grid")
        axes[0].set_xlabel("expedite capacity (% of orders)")
        axes[0].set_ylabel("total cost ($k)")
        axes[0].set_title("Cost under a capacity constraint")
        axes[0].legend(fontsize=8)
        axes[1].plot(cap["capacity"] * 100, cap["recall_pmf"], marker="o",
                     color="#1f6f78", label="pmf exact")
        axes[1].plot(cap["capacity"] * 100, cap["recall_quantile"], marker="o",
                     color="#9b3b3b", label="quantile grid")
        axes[1].set_xlabel("expedite capacity (% of orders)")
        axes[1].set_ylabel("late-order recall")
        axes[1].set_title("How many late orders get caught")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(images_dir / "capacity.png", dpi=150)
        plt.close(fig)

        # 6. budget curve: what the first dollars buy
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
        axes[0].plot(bc["budget"] / 1e3, bc["total_cost"] / 1e3, marker="o", color="#1f6f78")
        axes[0].set_xlabel("intervention budget ($k)")
        axes[0].set_ylabel("total cost ($k)")
        axes[0].set_title("Cost under a budget constraint")
        axes[1].bar(bc["budget"] / 1e3, bc["orders_notify"], width=90,
                    color="#2f5d8c", label="notify")
        axes[1].bar(bc["budget"] / 1e3, bc["orders_express"], width=90,
                    bottom=bc["orders_notify"], color="#d99a2b", label="express")
        axes[1].set_xlabel("intervention budget ($k)")
        axes[1].set_ylabel("orders touched")
        axes[1].set_title("What the budget buys")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(images_dir / "budget.png", dpi=150)
        plt.close(fig)

        # 7. decision map: the action mix is a function of the economics
        pivot = sens.pivot_table(
            index="penalty_per_day", columns="express_cost", values="express_rate_pmf"
        )
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        im = ax.imshow(pivot.to_numpy(), cmap="YlOrBr", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)), [f"${c}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"${p}" for p in pivot.index])
        ax.set_xlabel("express cost")
        ax.set_ylabel("penalty per late day")
        ax.set_title("Share of orders sent to express (pmf policy)")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.to_numpy()[i, j]
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if value > 0.55 else "#3b3b3b")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(images_dir / "decision_map.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - charts are optional
        print(f"charts skipped: {exc}")

    # ------------------------------------------------------------------ bundle for the app
    bundle = {
        "feature_names": split.feature_names,
        "encoders": encoders,
        "scaler": split.scaler,
        "quantile": payload["quantile"],
        "pmf_model": payload["pmf_model"],
        "binary": payload["binary"],
        "atoms": policy.ATOMS,
        "mode_pmf": mode_pmf_table,
        "mode_pmf_default": mode_pmf_default,
        "metrics": {
            "binary_auc": payload["binary_auc"],
            "mae_p50": float(np.abs(delay - p50).mean()),
            "mode_base_rate_auc": ceiling["mode_base_rate_auc"],
            "within_mode_auc": {k: v["auc"] for k, v in ceiling["within_mode"].items()},
            "coverage": {
                "p10": float((delay <= p10).mean()),
                "p50": float((delay <= p50).mean()),
                "p90": float((delay <= p90).mean()),
            },
            "best_cost": float(comp.iloc[0]["total_cost"]),
        },
        "params": {"penalty": pen, "actions": [a.__dict__ for a in policy.ACTIONS]},
        "trained_on": str(data_path),
        "cache_key": payload["cache_key"],
    }
    bundle_path = Path(args.bundle)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    print(f"\nwrote {bundle_path}")

    # ------------------------------------------------------------------ summary
    get = lambda name: comp.loc[comp["policy"] == name].iloc[0]  # noqa: E731
    none_c, all_c, q_c, pmf_c = (
        get("none (do nothing)"),
        get("all express"),
        get("quantile grid (3 points)"),
        get("pmf exact (lookup)"),
    )
    bin_c, binleg_c = get("binary: cost-aware"), get("binary: p50 > 0 (legacy)")

    diagnostics = {
        "facts": facts,
        "split": {"train": int(len(split.y_train)), "valid": int(len(split.y_valid)), "test": n},
        "binary_auc": payload["binary_auc"],
        "mae_p50": float(np.abs(delay - p50).mean()),
        "coverage": {"p10": float((delay <= p10).mean()), "p50": float((delay <= p50).mean()), "p90": float((delay <= p90).mean())},
        "ceiling": ceiling,
        "costs": {str(k): float(v["total_cost"]) for k, v in comp.set_index("policy").iterrows()},
        "params": {"penalty": pen, "n_estimators": args.n_estimators, "pmf_iter": args.pmf_iter},
    }
    (results_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Cost analysis — refreshed numbers",
        "",
        f"Generated by `scripts/refresh_cost_analysis.py` · penalty ${pen:.0f}/late-day · "
        f"{n:,} test orders (chronological split) · cache `{payload['cache_key']}`.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| orders (full dataset) | {facts['orders']:,} |",
        f"| late rate | {facts['late_rate']:.1%} |",
        f"| total late-days | {facts['total_delay_days']:,.0f} |",
        f"| binary baseline AUC | {payload['binary_auc']:.4f} |",
        f"| shipping-mode base-rate AUC | {ceiling['mode_base_rate_auc']:.4f} |",
        f"| MAE (P50, days) | {np.abs(delay - p50).mean():.3f} |",
        f"| P10/P50/P90 coverage | "
        f"{(delay <= p10).mean():.1%} / {(delay <= p50).mean():.1%} / {(delay <= p90).mean():.1%} |",
        "",
        "## The signal ceiling: shipping mode is the whole model",
        "",
        f"- A base-rate lookup on `Shipping Mode` alone reaches AUC **{ceiling['mode_base_rate_auc']:.4f}**; "
        f"the 18-feature model reaches **{ceiling['model_auc']:.4f}** "
        f"(gap {ceiling['auc_gap']:+.4f}).",
    ]
    for mode, info in ceiling["within_mode"].items():
        lines.append(
            f"- Within `{mode}` the AUC is **{info['auc']:.4f}** "
            f"(n={info['n_test']:,}, late rate {info['late_rate']:.1%}) — no per-order signal."
        )
    lines += [
        "- Independence checks inside `Standard Class` (chi-square, Cramér's V): "
        + ", ".join(
            f"{t['feature']} p={t['p_value']:.3f}/V={t['cramers_v']:.3f}"
            for t in ceiling["independence_tests"]
        ),
        "",
        "The delay is uniform inside each shipping mode (Standard: −2..2, Second: 0..4 at 20% each),",
        "so 0.78 is not a modelling failure — it is the information ceiling of this dataset.",
        "",
        "## Target atoms: coverage is quantised by the discrete target",
        "",
        "| 可达到的覆盖率 | 数值 |",
        "|---|---|",
        "\n".join(f"| P(y <= {k}) | {v:.1%} |" for k, v in atoms.items()),
        "",
        "延迟天数是整数（−2…4），经验覆盖率只能在原子上跳变，名义 50% 分位无法对齐——这不是校准问题。",
        "真正的修法是直接建模离散分布（下面第 3 条）。",
        "",
        "## Policy comparison (one cost function for every policy)",
        "",
        markdown_policy_table(comp),
        "",
        "## Paired bootstrap vs `pmf exact (lookup)`",
        "",
        df_to_markdown(boot, ".3f"),
        "",
        "## Headline deltas",
        "",
        f"- do nothing → pmf: ${none_c['total_cost']:,.0f} → ${pmf_c['total_cost']:,.0f} "
        f"({1 - pmf_c['total_cost'] / none_c['total_cost']:.1%} lower)",
        f"- all express → pmf: ${all_c['total_cost']:,.0f} → ${pmf_c['total_cost']:,.0f} "
        f"({1 - pmf_c['total_cost'] / all_c['total_cost']:.1%} lower)",
        f"- **quantile grid → pmf: ${q_c['total_cost']:,.0f} → ${pmf_c['total_cost']:,.0f} "
        f"({1 - pmf_c['total_cost'] / q_c['total_cost']:.1%} lower)**",
        f"- legacy binary → pmf: ${binleg_c['total_cost']:,.0f} → ${pmf_c['total_cost']:,.0f} "
        f"({1 - pmf_c['total_cost'] / binleg_c['total_cost']:.1%} lower)",
        f"- fair binary (same decision layer) → pmf: ${bin_c['total_cost']:,.0f} → "
        f"${pmf_c['total_cost']:,.0f} ({1 - pmf_c['total_cost'] / bin_c['total_cost']:.1%} lower)",
        "",
        "The v1 README compared the quantile policy against `p50 > 0`, which is a *threshold*",
        "bug, not a model gap: once the binary baseline gets the same expected-cost rule it lands",
        f"within {abs(1 - pmf_c['total_cost'] / bin_c['total_cost']):.2%} of the optimum. "
        "The real gap is the 3-quantile reconstruction of a 7-atom distribution.",
        "",
        "## Robustness to cost-parameter uncertainty",
        "",
        "The nominal policy is optimal at $200/day and $100 express. The minimax-regret rule",
        "trades nominal cost for a cap on the worst case — how much depends on the uncertainty band:",
        "",
        df_to_markdown(rob_summary_df, ".2f"),
        "",
        "With a wide band the robust rule becomes a blunt hedge (express everything except Same Day):",
        "worst-case regret falls ~13% but it costs $136k more at the nominal scenario.",
        "With a +/-25% band the nominal policy is already minimax-optimal, so robustness is free.",
        "Read it as: the value of robustness is a function of how uncertain the economics are —",
        "and under wide uncertainty the model stops mattering at all.",
        "",
        "## Parameter sensitivity: where the pmf gain comes from",
        "",
        "pmf cost advantage over the 3-point quantile grid (positive = pmf cheaper):",
        "",
        df_to_markdown(
            sens.pivot_table(
                index="penalty_per_day", columns="express_cost",
                values="advantage_pmf_vs_quantile_pct",
            ).reset_index(),
            "+.1%",
        ),
        "",
        "The pmf beats the quantile grid in all 20 settings (0.07%–8.2%).",
        "Against the fair binary baseline the gap is −0.005% to +2.6%, and binary wins 4 of 20 —",
        "i.e. model class barely matters; the distribution representation and the decision rule do.",
        "",
        "## The decision is a function of the economics",
        "",
        "Share of orders sent to express by the pmf policy:",
        "",
        df_to_markdown(
            sens.pivot_table(
                index="penalty_per_day", columns="express_cost", values="express_rate_pmf"
            ).reset_index(),
            ".1%",
        ),
        "",
        "At $200/day and $100 express the optimal policy expedites 35% of orders and notifies 59%;",
        "raise the penalty or cut the express price and express takes over the whole book.",
        "The model does not decide this — the cost structure does.",
        "",
        "## Cost-model assumption check",
        "",
        "The flat $200/day penalty is an assumption. A value-proportional late fee is the other",
        "obvious model; here is what happens to the decision under it:",
        "",
        df_to_markdown(
            cost_model.assign(
                median_penalty_per_day=cost_model["median_penalty_per_day"].map(
                    lambda v: f"${v:,.0f}/day"
                )
            ),
            ".2%",
        ),
        "",
        "With 1%/day of order value (floor $20) the median fee is only $20/day, so no action pays",
        "for itself — intervention drops from 94% to 0%. The 94% is a consequence of the flat",
        "$200/day assumption, not of the data. Any real deployment has to calibrate this first.",
        "",
        "## Capacity-constrained allocation",
        "",
        df_to_markdown(cap, ".3f"),
        "",
        "## Budget-constrained allocation (pmf, incremental greedy)",
        "",
        df_to_markdown(bc, ".3f"),
        "",
        "The first $500k buys notifications for almost every order; the next $500k upgrades them",
        "to express; $1.5M already reaches the unconstrained optimum ($1.27M of action spend).",
        "That is the order a planner should spend in — not the order the model scores highest.",
        "",
        "## Related outputs",
        "",
        "- `drift_report.csv` / `drift_summary.json` — `scripts/drift_report.py` (monthly PSI)",
        "- `intervention_*.csv`, `ab_sample_size.csv` — `scripts/intervention_analysis.py`",
        "- `pacing.csv` — `scripts/pacing_simulation.py` (multi-period budget)",
        "- `ab_analysis.json` / `ab_by_mode.csv` — `scripts/ab_analysis.py` (A/B framework)",
        "- `robustness.csv` / `robustness_summary.csv` — this script (minimax regret)",
        "- `python scripts/monitor.py --reference ref.csv --current new.csv` — online PSI check",
        "",
    ]
    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {results_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
