# Results

## Executive Summary

This project packages a GS-MS statistical-arbitrage experiment built around signature features and optimal stopping. In the current local reference run, the signature-based stopping strategy outperformed the moving-average baseline on total return (`5.93%` vs `2.04%`) while producing one additional trade (`5` vs `4`).

The current reference numbers come from the corrected normalized-spread pipeline, with the baseline now evaluated under paper-style capital-based pair accounting rather than the older spread-unit equity approximation. That change materially reduced the exaggerated return profile from earlier local runs and brought the baseline onto a scale closer to the paper.

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
- Baseline accounting: capital-based pair returns with position size set from current equity and the first leg price at entry
- Feature configuration: prefix signatures with depth `4`, scalar term enabled, time channel normalized on `[0, 1]`
- Stopping thresholds: `k = 0.05`, `mu = 20.0`

## Key Results

### Performance Summary

| Strategy | Trades | Win rate | Avg holding days | Sharpe | Max drawdown | Total net PnL | Final equity | Total return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SOT | 5 | 60.0% | 36.2 | 0.464 | -0.146 | 0.0593 | 1.0593 | 5.93% |
| Baseline | 4 | 50.0% | 6.75 | 0.265 | -0.099 | 0.0204 | 1.0204 | 2.04% |

### Current SOT Trade Ledger

| Trade | Entry date | Exit date | Holding days | Entry spread | Exit spread | Net PnL |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 2022-01-07 | 2022-01-21 | 9 | 0.1823 | 0.0435 | -0.1409 |
| 2 | 2022-01-27 | 2022-02-22 | 17 | 0.0029 | 0.0895 | 0.0845 |
| 3 | 2022-03-23 | 2022-09-06 | 114 | 0.0946 | 0.1509 | 0.0542 |
| 4 | 2022-09-15 | 2022-10-20 | 25 | 0.1295 | 0.2025 | 0.0708 |
| 5 | 2022-12-08 | 2023-01-03 | 16 | 0.2253 | 0.2182 | -0.0092 |

The fifth SOT trade was force-closed at the end of the trading horizon on `2023-01-03`, which is expected under the current engine configuration.

### Validation Checks

The latest reference backtest passed the built-in validation suite:

- SOT: no overlapping positions, correct event ordering, accounting identities
- Baseline: the same checks plus a baseline no-lookahead validation

## Generated Figures

Run the golden release pipeline and inspect these locally generated artifacts:

```powershell
docker compose run --rm quant python -m scripts.run_golden --run-id golden_release
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

After correcting the spread construction to use normalized prices and moving the baseline onto paper-style capital accounting, the strategy no longer exhibits the unrealistic baseline return inflation seen in the earlier spread-unit implementation. The updated comparison leaves the baseline much closer to the scale reported in the paper, even though it is not yet an exact match.

Relative to the baseline, SOT in the current run holds positions longer, completes one additional trade, and finishes with both higher total return and higher Sharpe. The baseline now reports materially lower PnL because it is measured as capital returns from position-sized pair trades rather than as compounded spread-unit changes. That contrast is visible in the equity and trade ledgers and is exactly why the repo keeps manifests, validation checks, and per-trade artifacts as first-class outputs.

The release-ready repo is therefore structured around reproducibility rather than around storing bulky outputs in Git. A clean clone can rebuild the full artifact tree locally and verify that the recorded metrics still line up with the published report.

## Reproduction Notes

- Build the image with `docker compose build`
- Run `docker compose run --rm quant python -m scripts.run_golden --run-id golden_release`
- Inspect `runs/golden_release/backtest/manifest.json` for hashes, seeds, versions, and artifact references
- Run `docker compose run --rm quant pytest` to verify the unit and smoke-test surface before publishing changes
