# Run Artifacts

This directory stores generated backtest outputs, model checkpoints, plots, manifests, and test artifacts.

When you run the project through Docker, the container writes directly back into the host `runs/` directory because the repository is bind-mounted into `/workspace`:

```powershell
docker compose run --rm quant python -m scripts.run_backtest --run-id golden_release
docker compose run --rm quant python -m scripts.run_golden --run-id golden_release
```

These outputs are intentionally ignored by Git so each clone can recreate them locally without bloating repository history.
