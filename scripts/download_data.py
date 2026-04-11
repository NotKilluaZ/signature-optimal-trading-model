import json
from pathlib import Path
import pandas as pd
from src.sigstop.config import load_config
from src.sigstop.data.providers import download_yfinance_symbol
from src.sigstop.paths import (
    RAW_DATA_DIR,
    MANIFESTS_DATA_DIR,
    ensure_directories,
    get_raw_provider_dir,
)

def main():
    config = load_config()

    data_cfg = config["data"]
    storage_cfg = config["storage"]

    symbols = data_cfg["symbols"]
    start_date = data_cfg["start_date"]
    end_date = data_cfg["end_date"]
    interval = data_cfg["interval"]
    auto_adjust = data_cfg["auto_adjust"]

    provider = data_cfg["provider"]
    overwrite = storage_cfg["overwrite"]

    ensure_directories()

    provider_dir = get_raw_provider_dir(provider)
    provider_dir.mkdir(parents = True, exist_ok = True)

    for symbol in symbols:
        print(f"\nDownloading data for {symbol}...")

        symbol_dir = provider_dir / symbol
        symbol_dir.mkdir(parents = True, exist_ok = True)

        file_path = symbol_dir / "daily.parquet"

        # Reuse the existing raw artifact when overwrite is disabled
        if file_path.exists() and not overwrite:
            df = pd.read_parquet(file_path)
            print(f"Reusing cached raw data at: {file_path}")
        else:
            df = download_yfinance_symbol(
                symbol = symbol,
                start_date = start_date,
                end_date = end_date,
                interval = interval,
                auto_adjust = auto_adjust,
            )

            df.to_parquet(file_path, index = False)

        manifest = {
            "symbol": symbol,
            "provider": provider,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "rows": len(df),
            "min_date": str(df["date"].min()),
            "max_date": str(df["date"].max()),
        }

        manifest_path = MANIFESTS_DATA_DIR / f"{provider}_{symbol}.json"

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent = 2)

        print(
            f"{symbol}: {len(df)} rows "
            f"({df['date'].min().date()} -> {df['date'].max().date()})"
        )


if __name__ == "__main__":
    main()
