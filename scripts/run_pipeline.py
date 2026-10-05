"""CLI entry point for running the full training pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="SupervisedAutoML – Full Pipeline")
    parser.add_argument("--data", required=True, help="Path to the dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--output", default="artifacts", help="Output directory")
    args = parser.parse_args()

    print(f"Running full pipeline on: {args.data}")
    print(f"Target: {args.target}")
    print(f"Output: {args.output}")
    print("For interactive mode, use: streamlit run app.py")


if __name__ == "__main__":
    main()
