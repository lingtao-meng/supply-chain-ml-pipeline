#!/usr/bin/env python3
"""Compare a new batch of orders against a reference window.

Exit codes make this cron/CI friendly:

* 0 — stable or moderate drift (monitor only)
* 1 — feature drift with a stable target (reweight or segment)
* 2 — target drift (retrain)

Usage::

    python scripts/monitor.py --reference train.csv --current new.csv
    python scripts/monitor.py --reference train.csv --current new.csv --out results/monitor.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import monitoring  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, help="CSV for the reference window")
    ap.add_argument("--current", required=True, help="CSV for the new batch")
    ap.add_argument("--target", default=monitoring.DEFAULT_TARGET)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    reference = pd.read_csv(args.reference, encoding="latin1")
    current = pd.read_csv(args.current, encoding="latin1")
    result = monitoring.monitor_frames(reference, current, target=args.target)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
