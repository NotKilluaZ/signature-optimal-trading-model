# Signature-Based Optimal Stopping for GS-MS Pairs Trading

![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB)
![Pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)
![License: MIT](https://img.shields.io/badge/license-MIT-2EA043)

This repository packages a reproducible research pipeline for a GS-MS pairs-trading experiment based on signature features and optimal stopping. It downloads public market data from yFinance, builds the spread, fits an OU generator, trains entry/exit stopping policies, and runs a side-by-side backtest against a rule-based baseline.

This project was developed based on the quant finance paper [Optimal Stopping via Signatures](https://arxiv.org/abs/2309.16008) (arXiv:2309.16008).

The repo is intentionally Git-clean by default: raw data, cached features, model checkpoints, and backtest runs are generated locally under `data/` and `runs/` and are excluded from version control.

## TL;DR

- Pair: Goldman Sachs (`GS`) and Morgan Stanley (`MS`)
- Pipeline: download data -> align -> split -> choose spread orientation/beta -> fit OU simulator -> train stopping policies -> backtest
- Reference local run: SOT total return `5.93%` on `5` trades vs baseline `2.04%` on `4` trades
- Scope: research prototype only, not investment advice

More detail lives in [Results.md](Results.md).

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/sigstop/` | Core pipeline code for data, spread construction, OU generation, features, stopping, training, and backtesting |
| `scripts/` | CLI entrypoints for each pipeline stage plus the one-shot golden run |
| `configs/default.yaml` | Default experiment configuration |
| `tests/` | Unit and smoke tests for the backtest and training surface |
| `data/README.md` | Notes on generated datasets and caches |
| `runs/README.md` | Notes on generated backtest outputs and manifests |
| `.github/workflows/ci.yml` | GitHub Actions workflow for install + test verification |

## Setup

Use Python 3.12

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .[dev]
```

If you are running CPU-only PyTorch locally, `torch>=2.11,<2.12` is intentionally left without a CUDA-specific wheel tag so the environment stays portable across GitHub Actions and local machines.

## Quick Start

Run the full release pipeline and write a stable artifact bundle:

```powershell
python -m scripts.run_golden --run-id golden_release
```

Run only the final backtest stage against existing cached artifacts:

```powershell
python -m scripts.run_backtest --run-id golden_release
```

Run the existing default end-to-end pipeline without a special run id:

```powershell
python -m scripts.run_pipeline
```

Run tests:

```powershell
pytest
```

## Reference Results

The current release docs summarize a corrected normalized-spread backtest on the default configuration in `configs/default.yaml`.

| Strategy | Trades | Avg holding days | Sharpe | Max drawdown | Total net PnL | Total return |
|    ---   |  ---:  |       ---:       |  ---:  |     ---:     |      ---:     |     ---:     |
|    SOT   |   5    |       36.2       | 0.464  |    -0.146    |     0.0593    |      5.93%   |
| Baseline |   4    |       6.75       | 0.265  |    -0.099    |     0.0204    |      2.04%   |

The baseline figures below use paper-style capital-based pair accounting rather than the older spread-unit equity approximation. Under that accounting, SOT finishes ahead on both total return and Sharpe in this reference run.

Key run facts from the same reference run:

- Formation window: `2021-01-04` to `2021-12-31`
- Trading window: `2022-01-03` to `2023-01-03`
- Price basis: normalized full aligned history before beta search and spread construction
- Selected spread orientation: `GS_minus_beta_MS`
- Selected hedge ratio (`beta`): `0.862099`
- Baseline accounting: paper-style capital returns from position-sized GS/MS pair trades
- Signature depth: `4`
- Deterministic threshold: `k = 0.05`, `mu = 20.0`

Because generated artifacts are ignored, figures and manifests are recreated locally when you run the pipeline rather than stored in Git. The expected output locations are documented in [Results.md](Results.md), [data/README.md](data/README.md), and [runs/README.md](runs/README.md).

## What the Golden Run Produces

After `python -m scripts.run_golden --run-id golden_release`, you should expect:

- `data/processed/artifacts/plots/gs_ms_normalized_prices.png`
- `data/processed/artifacts/plots/gs_ms_spread_trading.png`
- `runs/golden_release/backtest/summary_table.csv`
- `runs/golden_release/backtest/manifest.json`
- `runs/golden_release/backtest/plots/equity_curve_comparison.png`
- `runs/golden_release/backtest/sot/trades.csv`
- `runs/golden_release/backtest/baseline/trades.csv`

## Limitations and Next Steps

- The public release is scoped to the GS-MS experiment; spread construction is still hard-coded to that pair.
- The repo is a research prototype and does not support live trading or broker integration.
- Hyperparameter sweeps and richer calibration studies are still manual rather than automated report-generation steps.
- The README documents one reference run; if the strategy logic changes, rerun `scripts.run_golden` and refresh `Results.md`.

## License, Citation, and Disclaimer

This project is released under the [MIT License](LICENSE).

If you reference the work, cite both this repository and the original signature-based optimal stopping paper that inspired the experiment. This code uses only historical public market data and is intended for research and educational use only. It is not investment advice and should not be used to trade real capital without substantial additional validation.
