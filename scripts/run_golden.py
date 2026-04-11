from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.run_pipeline import PIPELINE_STEPS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_release_commands(
    *,
    run_id: str,
    output_dir: str | None = None,
) -> list[list[str]]:
    commands = [[sys.executable, "-m", step] for step in PIPELINE_STEPS[:-1]]

    backtest_command = [
        sys.executable,
        "-m",
        PIPELINE_STEPS[-1],
        "--run-id",
        run_id,
    ]
    if output_dir is not None:
        backtest_command.extend(["--output-dir", output_dir])

    commands.append(backtest_command)
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description = "Run the full release pipeline and write a stable golden-run artifact set.",
    )
    parser.add_argument(
        "--run-id",
        type = str,
        default = "golden_release",
        help = "Run identifier used for the final backtest artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        type = str,
        default = None,
        help = "Optional explicit output directory for the final backtest stage.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    commands = build_release_commands(run_id = args.run_id, output_dir = args.output_dir)

    print("Running golden release pipeline...\n")
    for step, command in zip(PIPELINE_STEPS, commands):
        print(f"=== Running {step} ===")
        subprocess.run(command, cwd = PROJECT_ROOT, check = True)
        print(f"=== Completed {step} ===\n")

    print("Golden release pipeline finished successfully.")


if __name__ == "__main__":
    main()
