"""Find exact coverage valuations at which a sequential budget policy changes rank."""

import csv
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ONLINE = ("first-come", "historical threshold", "daily pacing")


def optimal_ranges(options):
    """Each option maps to (simulated cost, number of unattended priority orders)."""
    for name, (cost, missed) in options.items():
        lower, upper = Fraction(0), None
        for other_cost, other_missed in options.values():
            slope, gap = missed - other_missed, other_cost - cost
            if slope > 0:
                bound = Fraction(gap, slope)
                upper = bound if upper is None else min(bound, upper)
            elif slope < 0:
                lower = max(lower, Fraction(gap, slope))
            elif gap < 0:
                break
        else:
            if upper is None or lower <= upper:
                yield name, lower, upper


def compare(folder):
    with (folder / "arrival_pacing.csv").open() as stream:
        summaries = {(int(r["total_budget"]), r["strategy"]): r for r in csv.DictReader(stream)}
    daily = {}
    with (folder / "arrival_pacing_by_day.csv").open() as stream:
        for r in csv.DictReader(stream):
            key = (int(r["total_budget"]), r["strategy"])
            if key[1] in ONLINE:
                cost, missed = daily.get(key, (0, 0))
                daily[key] = (cost + Fraction(r["cost"]), missed + int(r["priority_no_action"]))
    for budget in sorted({k[0] for k in summaries}):
        options = {name: daily[budget, name] for name in ONLINE}
        for name, (cost, missed) in options.items():
            row = summaries[budget, name]
            if (cost, missed) != (Fraction(row["cost"]), int(row["priority_no_action"])):
                raise ValueError(f"Daily totals differ for {budget}, {name}")
        for name, lower, upper in optimal_ranges(options):
            yield dict(budget=budget, strategy=name, cost=options[name][0],
                       missed_priority=options[name][1], lower_inclusive=str(lower),
                       upper_inclusive="" if upper is None else str(upper))


if __name__ == "__main__":
    folder = ROOT / "results"
    rows = list(compare(folder))
    with (folder / "priority_tradeoff.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
