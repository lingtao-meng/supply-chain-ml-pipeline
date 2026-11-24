"""Plot the preferred strategy and its switch point at each action budget."""

from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
INK, GRAY = "#1b2c38", "#4e606d"
BLUE, ORANGE = "#236b8e", "#a64b30"



def configure_style():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "STHeiti", "PingFang HK")
    family = next((name for name in candidates if name in installed), None)
    if family is None:
        raise RuntimeError("Install Noto Sans CJK SC to render the Chinese labels")
    plt.rcParams.update({
        "font.family": family, "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": "#aebac2", "axes.linewidth": .75,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": GRAY, "ytick.color": GRAY, "axes.unicode_minus": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "grid.color": "#e5eaee", "grid.linewidth": .7,
    })


def main():
    configure_style()
    table = pd.read_csv(ROOT / "results/priority_tradeoff.csv")
    historical = table[table.strategy == "historical threshold"].set_index("budget").sort_index()
    daily = table[table.strategy == "daily pacing"].set_index("budget").loc[historical.index]
    cutoff = historical.lower_inclusive.map(lambda value: float(Fraction(value)))
    additional_orders = daily.missed_priority - historical.missed_priority
    additional_cost = historical.cost - daily.cost

    right_edge = 250
    fig = plt.figure(figsize=(9.4, 4.0))
    grid = fig.add_gridspec(1, 2, width_ratios=(5.1, 2.3), left=.12, right=.97,
                           bottom=.20, top=.79, wspace=.22)
    chart = fig.add_subplot(grid[0])
    ledger = fig.add_subplot(grid[1], sharey=chart)
    for row, budget in enumerate(cutoff.index):
        y = len(cutoff) - row - 1
        switch = cutoff[budget]
        chart.hlines(y, 0, switch, color=BLUE, linewidth=7)
        chart.hlines(y, switch, right_edge, color=ORANGE, linewidth=7)
        chart.scatter(switch, y, s=40, edgecolor=INK, facecolor="white", linewidth=1.15, zorder=4)
        chart.annotate(f"{switch:.2f}", (switch, y), xytext=(0, 8),
                       textcoords="offset points", ha="center", va="bottom")
        ledger.text(.31, y, f"+{additional_orders[budget]:,}", ha="right", va="center")
        ledger.text(.98, y, f"+{additional_cost[budget]:,.0f}", ha="right", va="center")
        ledger.axhline(y-.42, color="#e5eaee", linewidth=.65)
    chart.set(xlim=(0, right_edge), ylim=(-.48, len(cutoff)-.43), xticks=range(0, 251, 50),
              yticks=range(len(cutoff)), yticklabels=[str(b//10000) for b in cutoff.index[::-1]],
              xlabel="每多保障一单的额外估值（美元/单）", ylabel="预算（万美元）")
    chart.set_axisbelow(True)
    chart.grid(axis="x")
    chart.spines["left"].set_visible(False)
    chart.tick_params(axis="y", length=0, pad=8)
    ledger.set_axis_off()
    ledger.set_xlim(0, 1)
    ledger.text(.98, 1.08, "历史阈值较按日释放", transform=ledger.transAxes,
                ha="right", va="center", color=INK, fontsize=11)
    for x, label in ((.31, "多覆盖（单）"), (.98, "额外成本（美元）")):
        ledger.text(x, 1.0, label, transform=ledger.transAxes,
                    ha="right", va="center", color=GRAY, fontsize=10)
    handles = [
        Line2D([], [], color=BLUE, linewidth=4, label="按日释放预算"),
        Line2D([], [], color=ORANGE, linewidth=4, label="按历史阈值干预"),
        Line2D([], [], color=INK, marker="o", markerfacecolor="white", ls="", label="策略分界"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.119, .98),
               ncol=3, borderaxespad=0, fontsize=10)
    fig.savefig(ROOT / "images/priority_tradeoff.png", dpi=210)
    plt.close(fig)


if __name__ == "__main__":
    main()
