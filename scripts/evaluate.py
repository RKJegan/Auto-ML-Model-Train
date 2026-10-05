"""CLI entry point for model evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="SupervisedAutoML – Evaluate")
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--data", required=True, help="Path to evaluation dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    args = parser.parse_args()

    print(f"Evaluating model: {args.model}")
    print(f"On dataset: {args.data}")


if __name__ == "__main__":
    main()
