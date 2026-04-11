from __future__ import annotations

from scripts.run_golden import build_release_commands
from scripts.run_pipeline import PIPELINE_STEPS


def test_run_pipeline_uses_full_backtest_by_default() -> None:
    assert "scripts.run_backtest" in PIPELINE_STEPS
    assert "scripts.run_stopping_policy" not in PIPELINE_STEPS
    assert "scripts.build_features" not in PIPELINE_STEPS
    assert PIPELINE_STEPS[-1] == "scripts.run_backtest"


def test_run_golden_routes_run_id_and_output_dir_to_backtest_command() -> None:
    commands = build_release_commands(
        run_id = "golden_release",
        output_dir = "runs/golden_release/backtest",
    )

    assert len(commands) == len(PIPELINE_STEPS)
    assert [command[2] for command in commands[:-1]] == PIPELINE_STEPS[:-1]
    assert commands[-1][2:] == [
        "scripts.run_backtest",
        "--run-id",
        "golden_release",
        "--output-dir",
        "runs/golden_release/backtest",
    ]
