# Generated Data

This directory is intentionally kept out of Git history.

The pipeline recreates the following locally:

- `data/raw/`: downloaded market data from `yfinance`
- `data/processed/`: aligned prices, formation/trading splits, spread artifacts, and plots
- `data/features/`: cached signature feature tensors and manifests
- `data/synthetic/`: cached OU simulation batches and validation reports
- `data/manifests/`: download manifests and other reproducibility metadata

Typical workflow:

```powershell
python -m scripts.run_golden --run-id golden_release
```

If you want to refresh only the source data, run:

```powershell
python -m scripts.download_data
```
