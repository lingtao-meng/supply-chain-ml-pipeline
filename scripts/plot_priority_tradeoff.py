"""Plot the saved experimental results for the project report."""

from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]


def configure_font():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    families = ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "STHeiti", "PingFang HK")
    font = next((name for name in families if name in installed), None)
    if font is None:
        raise RuntimeError("Install Noto Sans CJK SC to render the Chinese labels")
    plt.rcParams.update({"font.family": font, "font.size": 10.5, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "legend.frameon": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    configure_font()
    table = pd.read_csv(ROOT / "results/priority_tradeoff.csv")
    threshold = table.loc[table.strategy == "historical threshold"].sort_values("budget")
    values = [float(Fraction(value)) for value in threshold.lower_inclusive]
    fig, ax = plt.subplots(figsize=(7.2, 3.45), layout="constrained")
    bars = ax.barh([str(b // 10000) for b in threshold.budget], values,
                   color="#41677d", height=.46)
    ax.bar_label(bars, fmt="%.2f", padding=5, fontsize=10)
    ax.set(xlabel="每多覆盖一笔高优先级订单的额外成本（美元/单）", ylabel="动作预算（万美元）",
           xlim=(0, 250), xticks=range(0, 251, 50))
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=".88", lw=.6)
    fig.savefig(ROOT / "images/priority_tradeoff.png", dpi=200)

    plt.close(fig)


if __name__ == "__main__":
    main()
