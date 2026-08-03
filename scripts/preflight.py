from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sumoe.config import load_config
from sumoe.training.preflight import run_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the exact SUMoE training environment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    report = run_preflight(
        load_config(args.config, args.overlay), check_only=args.check_only
    )
    print(json.dumps({"checks": report.checks, "errors": report.errors}, indent=2))


if __name__ == "__main__":
    main()
