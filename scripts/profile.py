"""CLI entry point for dataset profiling."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="SupervisedAutoML – Profile Dataset")
    parser.add_argument("--data", required=True, help="Path to the dataset")
    args = parser.parse_args()

    print(f"Profiling dataset: {args.data}")


if __name__ == "__main__":
    main()
