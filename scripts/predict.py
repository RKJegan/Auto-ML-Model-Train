"""CLI entry point for prediction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="SupervisedAutoML – Predict")
    parser.add_argument("--model", required=True, help="Path to the trained model artifact")
    parser.add_argument("--data", required=True, help="Path to input data for prediction")
    parser.add_argument("--output", default=None, help="Path to save predictions")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Input data: {args.data}")
    print("Use the Streamlit app for interactive prediction: streamlit run app.py")


if __name__ == "__main__":
    main()
