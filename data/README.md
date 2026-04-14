# Data Artifacts

This directory is where the project writes raw downloads, processed datasets, feature caches, manifests, and synthetic OU samples.

When you run the Docker workflow, the repository is bind-mounted into the container at `/workspace`, so files created under `data/` remain on the host machine:

```powershell
docker compose run --rm quant
docker compose run --rm quant python -m scripts.download_data
```

These generated artifacts are intentionally ignored by Git. Delete or refresh them locally as needed when you want to rebuild the pipeline outputs from scratch.
