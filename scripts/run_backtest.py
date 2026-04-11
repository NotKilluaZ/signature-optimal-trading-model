from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sigstop.backtest.runner import run_full_backtest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Run the GS-MS backtest artifact pipeline.")
    parser.add_argument(
        "--config",
        type = str,
        default = None,
        help = "Optional path to a YAML config file.",
    )
    parser.add_argument(
        "--run-id",
        type = str,
        default = "full_backtest",
        help = "Run identifier used under the artifacts root.",
    )
    parser.add_argument(
        "--output-dir",
        type = str,
        default = None,
        help = "Optional explicit output directory for backtest artifacts.",
    )
    return parser


# Run the default full backtest entrypoint from the command line
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_full_backtest(
        config_path = args.config,
        run_id = args.run_id,
        output_dir = args.output_dir,
    )

    print("\nBacktest run complete.")
    print(f"Run id: {result.run_id}")
    print(f"Output dir: {result.output_dir}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Summary table: {result.summary_table_path}")


if __name__ == "__main__":
    main()
