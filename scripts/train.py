"""CLI entry point for model training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="SupervisedAutoML – Train models")
    parser.add_argument("--data", required=True, help="Path to the training dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--output", default="artifacts/models", help="Output directory for trained models")
    args = parser.parse_args()

    print(f"Training on: {args.data}")
    print(f"Target column: {args.target}")
    print(f"Output dir: {args.output}")
    print("Use the Streamlit app for full interactive training: streamlit run app.py")


if __name__ == "__main__":
    main()
