#!/usr/bin/env python3
"""Check saved analysis tables and update the result summary without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- results:start -->", "<!-- results:end -->"


def close(actual: float, expected: float, label: str) -> None:
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-7):
        raise ValueError(f"{label}: {actual} != {expected}")


def build(root: Path = ROOT) -> tuple[str, str]:
    paths = ["results/diagnostics.json", "results/policy_comparison.csv", "results/mode_table.csv",
             "results_asof/diagnostics.json"]
    d = json.loads((root / paths[0]).read_text(encoding="utf-8"))
    with (root / paths[1]).open(encoding="utf-8", newline="") as f:
        policies = list(csv.DictReader(f))
    by_name = {r["policy"]: r for r in policies}
    if len(by_name) != len(policies) or set(by_name) != set(d["costs"]):
        raise ValueError("Policy names must be unique and match diagnostics.json")
    n = int(d["split"]["test"])
    for row in policies:
        name = row["policy"]
        total = float(row["total_cost"])
        close(total, float(d["costs"][name]), f"{name}: total cost")
        close(float(row["cost_per_order"]), total / n, f"{name}: cost per order")
        touched = int(row["orders_touched"])
        if touched != int(row["orders_notify"]) + int(row["orders_express"]) or not 0 <= touched <= n:
            raise ValueError(f"{name}: inconsistent action counts")
        close(float(row["intervention_rate"]), touched / n, f"{name}: intervention rate")
    lower = float(by_name["oracle (hindsight)"]["total_cost"])
    if any(float(row["total_cost"]) < lower for row in policies):
        raise ValueError("A feasible strategy cannot beat the hindsight cost lower bound")
    if int(by_name["capacity 25% (best-first)"]["orders_touched"]) > math.floor(n * 0.25):
        raise ValueError("Capacity exceeds floor(25% * test orders)")
    budget = by_name["budget $500k (pmf)"]
    spend = int(budget["orders_notify"]) * 20 + int(budget["orders_express"]) * 100
    if spend > 500000:
        raise ValueError("Action spending exceeds the stated budget")
    with (root / paths[2]).open(encoding="utf-8", newline="") as f:
        close(sum(int(r["orders"]) for r in csv.DictReader(f)), d["split"]["train"], "Training lookup count")
    asof = json.loads((root / paths[3]).read_text(encoding="utf-8"))
    close(asof["split"]["test"], n, "as-of test sample count")
    lookup = float(by_name["pmf exact (lookup)"]["total_cost"])
    express = float(by_name["all express"]["total_cost"])
    improvement = 100 * (express - lookup) / express
    summary = (
        f"- DataCo：{d['facts']['orders']:,} 个订单，测试集 {n:,} 单。\n"
        f"- 二分类 AUC：{d['binary_auc']:.4f}；运输方式查表基准：{d['ceiling']['mode_base_rate_auc']:.4f}。\n"
        f"- 给定动作与费用参数，历史分布查表策略成本 ${lookup:,.0f}，"
        f"较全量加急 ${express:,.0f} 低 **{improvement:.2f}%**。\n"
        f"- 这是统一成本函数下的样本外离线比较；历史分布查表与机器学习模型分别报告。"
    )
    lines = ["# 交付分析结果核对", "", "由 `python scripts/verify_results.py --write` 生成。",
             "核对已保存表格的内部一致性，不执行模型训练，也不代替数据到模型的完整复现。", "", summary, "",
             "## 统计口径", "", f"订单级累计正延迟：{d['facts']['total_delay_days']:,.0f} 天。",
             f"默认训练/验证/测试：{d['split']['train']:,} / {d['split']['valid']:,} / {n:,}。",
             f"严格 as-of：{asof['split']['train']:,} / {asof['split']['valid']:,} / {asof['split']['test']:,}。",
             "默认回放按订单日期切分；as-of 另剔除边界上标签尚未产生的样本。", "",
             "## 统一成本比较", "", "| 策略 | 总成本（美元） | 单均成本（美元/单） | 通知/加急 |",
             "|---|---:|---:|---:|"]
    for row in policies:
        name = row['policy'] + ("：成本下界" if row['policy'] == 'oracle (hindsight)' else "")
        lines.append(f"| {name} | {float(row['total_cost']):,.0f} | {float(row['cost_per_order']):.2f} | {row['orders_notify']} / {row['orders_express']} |")
    lines += ["", f"预算策略动作支出 ${spend:,.0f}，未达到 $500,000 上限；预算在该设定下不构成约束。",
              "", "## 核对范围", "", "检查策略名称、JSON/CSV成本、总额与单均值、动作数量、干预率、预算、容量、查表样本数及成本下界。",
              "单笔反事实成本、模型训练和模拟实验需使用各自正式命令另行复现。", "", "## 输入文件 SHA-256", ""]
    for path in paths:
        lines.append(f"- `{path}`：`{hashlib.sha256((root / path).read_bytes()).hexdigest()}`")
    return summary, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Update README result block and docs/RESULTS.md")
    args = parser.parse_args()
    summary, report = build()
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    updated = before + START + "\n" + summary + "\n" + END + after
    output = ROOT / "docs/RESULTS.md"
    if args.write:
        output.parent.mkdir(exist_ok=True)
        output.write_text(report, encoding="utf-8")
        readme.write_text(updated, encoding="utf-8")
    elif text != updated or not output.exists() or output.read_text(encoding="utf-8") != report:
        raise SystemExit("Result summary is stale. Run python scripts/verify_results.py --write")
    print("Saved result tables and report are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
