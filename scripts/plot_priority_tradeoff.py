"""Incremental coverage, cost and break-even value on the held-out period."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figure_style import format_figure  # noqa: E402


def main():
    records = pd.read_csv(ROOT / "results/priority_tradeoff.csv")
    by_policy = records.pivot(index="budget", columns="strategy", values=["cost", "missed_priority"])
    cost = by_policy["cost"]["historical threshold"] - by_policy["cost"]["daily pacing"]
    orders = by_policy["missed_priority"]["daily pacing"] - by_policy["missed_priority"]["historical threshold"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.3, 3.45), layout="constrained")
    for budget, extra in orders.items():
        dollars = cost.loc[budget]
        left.plot([0, extra], [0, dollars / 1000], lw=.85, color="#a4adb3")
        left.plot(extra, dollars / 1000, "o", color="#244e68", ms=4)
        left.annotate(f"{budget // 10000} 万", (extra, dollars / 1000),
                      xytext=(2, 6), textcoords="offset points", ha="center", fontsize=9)
    left.set(xlabel="增加覆盖的优先订单（单）", ylabel="增加的模拟成本（千美元）",
             xlim=(-20, 770), ylim=(-2, 99))
    left.set_title("A    成本与覆盖", loc="left")
    threshold = cost / orders
    for budget, value in threshold.items():
        y = budget / 10000
        right.hlines(y, 0, value, color="#a4adb3", lw=.8)
        right.plot(value, y, "o", mfc="white", mec="#244e68", ms=5)
        right.annotate(f"{value:.2f}", (value, y), xytext=(6, 0),
                       textcoords="offset points", va="center", fontsize=9)
    right.set(xlabel="两种策略持平时的估值（美元/单）", ylabel="干预预算（万美元）",
              yticks=threshold.index / 10000, ylim=(5, 45), xlim=(0, 265))
    right.set_title("B    每多保障一单的成本", loc="left")
    for ax in (left, right):
        ax.spines[["top","right"]].set_visible(False)
    format_figure(fig)
    fig.savefig(ROOT / "images/priority_tradeoff.png", dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    main()
