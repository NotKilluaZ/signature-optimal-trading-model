# Generated Runs

This directory holds local experiment outputs and is intentionally excluded from version control.

The main release workflow writes artifacts such as:

- `runs/<run-id>/backtest/manifest.json`
- `runs/<run-id>/backtest/summary_table.csv`
- `runs/<run-id>/backtest/plots/*.png`
- `runs/<run-id>/backtest/sot/`
- `runs/<run-id>/backtest/baseline/`

To regenerate the default release bundle:

```powershell
python -m scripts.run_golden --run-id golden_release
```

The backtest manifest records config hashes, input hashes, environment versions, and references to model artifacts so a reviewer can reproduce the same run locally.
