"""Write benchmark results. Full runs are manual; CI uses --quick."""

from __future__ import annotations

import argparse
from pathlib import Path

from jev_xai.bench import write_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/latest.json"))
    args = parser.parse_args()
    payload = write_benchmarks(args.out, quick=args.quick)
    print(f"wrote {args.out} ({len(payload['tasks'])} tasks)")


if __name__ == "__main__":
    main()
