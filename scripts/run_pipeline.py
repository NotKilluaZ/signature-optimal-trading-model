from __future__ import annotations
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_STEPS = [
    # Download stock pair data from yfinance
    "scripts.download_data",
    # Align their data frames
    "scripts.align_data",
    # Split into 252 formation & 252 trading 
    "scripts.make_split",
    # Build spread between stock pair. Calculate spread for both orientations and choose the better one. Choose beta value by maximizing OU likelihood on formation split
    "scripts.build_spread",
    # Generate random OU sample paths by choosing OU parameters fitted on formation spread from transition-density MLE then simulates paths with those OU parameter values
    "scripts.build_synthetic_ou",
    # Run the full backtest over the 252 day trading window
    "scripts.run_backtest",
]

# Run this command from project root, with venv active:   python scripts/run_pipeline.py
def main() -> None:
    print("Running quant pipeline...\n")

    for step in PIPELINE_STEPS:
        print(f"=== Running {step} ===")

        subprocess.run(
            [sys.executable, "-m", step],
            cwd=PROJECT_ROOT,
            check=True,
        )

        print(f"=== Completed {step} ===\n")

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
