# Results

## Executive Summary

This project packages a GS-MS statistical-arbitrage experiment built around signature features and optimal stopping. In the current local reference run, the signature-based stopping strategy outperformed the moving-average baseline on total return (`22.39%` vs `5.24%`) while producing one additional trade (`5` vs `4`).

The current reference numbers come from the corrected normalized-spread pipeline rather than the older raw-price spread construction. That change materially reduced the exaggerated return profile from earlier local runs and brought the backtest onto a more realistic scale.

The repo is deliberately checked in without raw data, checkpoints, or plots so Git history stays focused on the pipeline itself. Re-running the golden pipeline regenerates the figures, manifests, trade ledgers, and model artifacts locally.

## Dataset and Method

- Instruments: Goldman Sachs (`GS`) and Morgan Stanley (`MS`)
- Source: `yfinance`
- Full download window: `2021-01-01` to `2023-01-31`
- Formation period: `252` business days
- Trading period: `252` business days
- Price basis: normalize both series by their first aligned observation over the full history before beta search and spread construction
- Spread selection: evaluate `GS_minus_beta_MS` and `MS_minus_beta_GS`, choose the higher OU likelihood on the normalized formation window
- Selected orientation in the reference run: `GS_minus_beta_MS`
- Selected hedge ratio in the reference run: `beta = 0.862099`
- Feature configuration: prefix signatures with depth `4`, scalar term enabled, time channel normalized on `[0, 1]`
- Stopping thresholds: `k = 0.05`, `mu = 20.0`

## Key Results

### Performance Summary

| Strategy | Trades | Win rate | Avg holding days | Sharpe | Max drawdown | Total net PnL | Final equity | Total return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SOT | 5 | 80.0% | 24.4 | 1.298 | -0.044 | 0.2239 | 1.2239 | 22.39% |
| Baseline | 4 | 100.0% | 6.75 | 1.820 | 0.000 | 0.0524 | 1.0524 | 5.24% |

### Representative SOT Trade Ledger

| Trade | Entry date | Exit date | Holding days | Entry spread | Exit spread | Net PnL |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 2022-01-31 | 2022-03-17 | 32 | 0.0281 | 0.1050 | 0.0748 |
| 2 | 2022-04-06 | 2022-05-05 | 20 | 0.1402 | 0.0949 | -0.0475 |
| 3 | 2022-06-01 | 2022-07-27 | 38 | 0.1356 | 0.1787 | 0.0410 |
| 4 | 2022-10-10 | 2022-11-08 | 21 | 0.1383 | 0.2849 | 0.1444 |
| 5 | 2022-12-15 | 2023-01-03 | 11 | 0.2048 | 0.2182 | 0.0112 |

The fifth SOT trade was force-closed at the end of the trading horizon on `2023-01-03`, which is expected under the current engine configuration.

### Validation Checks

The latest reference backtest passed the built-in validation suite:

- SOT: no overlapping positions, correct event ordering, accounting identities
- Baseline: the same checks plus a baseline no-lookahead validation

## Generated Figures

Run the golden release pipeline and inspect these locally generated artifacts:

```powershell
python -m scripts.run_golden --run-id golden_release
```

Expected figure paths:

- `data/processed/artifacts/plots/gs_ms_normalized_prices.png`
- `data/processed/artifacts/plots/gs_ms_spread_trading.png`
- `runs/golden_release/backtest/plots/equity_curve_comparison.png`
- `runs/golden_release/backtest/plots/drawdown_comparison.png`
- `runs/golden_release/backtest/plots/holding_time_histogram.png`
- `runs/golden_release/backtest/plots/spread_with_trade_markers.png`

These cover the release checklist items for normalized prices, spread with entry/exit markers, equity-curve comparison, and holding-time diagnostics. The generated run directory also includes `summary_table.csv`, per-strategy `trades.csv`, and `manifest.json`.

## Interpretation

After correcting the spread construction to use normalized prices, the strategy no longer exhibits the unrealistic triple-digit returns seen in the earlier raw-price reference run. The updated backtest is still favorable to SOT on total PnL and total return, but it now presents a more balanced trade-off.

Relative to the baseline, SOT in the current run holds positions longer, completes one additional trade, and finishes with materially higher total return. The baseline remains smoother on a risk-adjusted basis in this sample, with higher Sharpe and zero drawdown. That contrast is visible in the equity and trade ledgers and is exactly why the repo keeps manifests, validation checks, and per-trade artifacts as first-class outputs.

The release-ready repo is therefore structured around reproducibility rather than around storing bulky outputs in Git. A clean clone can rebuild the full artifact tree locally and verify that the recorded metrics still line up with the published report.

## Reproduction Notes

- Install with `pip install -r requirements.txt` and `pip install -e .[dev]`
- Run `python -m scripts.run_golden --run-id golden_release`
- Inspect `runs/golden_release/backtest/manifest.json` for hashes, seeds, versions, and artifact references
- Run `pytest` to verify the unit and smoke-test surface before publishing changes
