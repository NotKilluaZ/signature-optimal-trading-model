import json
import pandas as pd
from pathlib import Path
from src.sigstop.config import load_config
from src.sigstop.paths import PROCESSED_DATA_DIR
from src.sigstop.data.split import create_formation_trading_split

def main():
    config = load_config()

    aligned_path = PROCESSED_DATA_DIR / "aligned_prices.parquet"

    df = pd.read_parquet(aligned_path)

    split_cfg = config["split"]
    formation_days = int(split_cfg["formation_days"])
    trading_days = int(split_cfg["trading_days"])

    formation, trading, metadata = create_formation_trading_split(
        df,
        formation_days = formation_days,
        trading_days = trading_days,
    )

    formation_path = PROCESSED_DATA_DIR / "formation.parquet"
    trading_path = PROCESSED_DATA_DIR / "trading.parquet"
    metadata_path = PROCESSED_DATA_DIR / "split_metadata.json"

    formation.to_parquet(formation_path, index = False)
    trading.to_parquet(trading_path, index = False)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent = 2)

    print("\nSplit complete:")
    print(metadata)

if __name__ == "__main__":
    main()
