# Results

## Executive Summary

This project packages a GS-MS statistical-arbitrage experiment built around signature features and optimal stopping. In the current local reference run, the signature-based stopping strategy outperformed the moving-average baseline on annualized Sharpe (`0.949` vs `0.474`) while producing the same number of trades (`4`).

The repo is deliberately checked in without raw data, checkpoints, or plots so Git history stays focused on the pipeline itself. Re-running the golden pipeline regenerates the figures, manifests, trade ledgers, and model artifacts locally.

## Dataset and Method

- Instruments: Goldman Sachs (`GS`) and Morgan Stanley (`MS`)
- Source: `yfinance`
- Full download window: `2021-01-01` to `2023-01-31`
- Formation period: `252` business days
- Trading period: `252` business days
- Spread selection: evaluate `GS_minus_beta_MS` and `MS_minus_beta_GS`, choose the higher OU likelihood
- Selected orientation in the reference run: `MS_minus_beta_GS`
- Selected hedge ratio in the reference run: `beta = 0.217523`
- Feature configuration: prefix signatures with depth `4`, scalar term enabled, time channel normalized on `[0, 1]`
- Stopping thresholds: `k = 0.05`, `mu = 20.0`

## Key Results

### Performance Summary

| Strategy | Trades | Win rate | Avg holding days | Sharpe | Max drawdown | Total net PnL | Final equity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SOT | 4 | 50.0% | 9.5 | 0.949 | -0.876 | 1.2696 | 2.2696 |
| Baseline | 4 | 50.0% | 28.0 | 0.474 | -0.502 | 0.0811 | 1.0811 |

### Representative SOT Trade Ledger

| Trade | Entry date | Exit date | Holding days | Entry spread | Exit spread | Net PnL |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 2022-01-28 | 2022-01-31 | 1 | 21.1369 | 20.2841 | -0.8755 |
| 2 | 2022-10-31 | 2022-12-19 | 34 | 5.1530 | 7.7198 | 2.5584 |
| 3 | 2022-12-22 | 2022-12-27 | 2 | 8.2620 | 7.8093 | -0.4628 |
| 4 | 2022-12-30 | 2023-01-03 | 1 | 7.4904 | 7.5495 | 0.0496 |

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

The current signature-based strategy is materially more aggressive than the baseline: it exits faster, realizes far higher total PnL in the reference run, and improves Sharpe, but it also experiences a deeper drawdown profile. That trade-off is visible in the comparison metrics and is exactly why the repo keeps the backtest manifest, per-trade ledgers, and validation checks as first-class artifacts.

The release-ready repo is therefore structured around reproducibility rather than around storing bulky outputs in Git. A clean clone can rebuild the full artifact tree locally and verify that the recorded metrics still line up with the published report.

## Reproduction Notes

- Install with `pip install -r requirements.txt` and `pip install -e .[dev]`
- Run `python -m scripts.run_golden --run-id golden_release`
- Inspect `runs/golden_release/backtest/manifest.json` for hashes, seeds, versions, and artifact references
- Run `pytest` to verify the unit and smoke-test surface before publishing changes
