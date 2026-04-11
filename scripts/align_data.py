from pathlib import Path
from src.sigstop.config import load_config
from src.sigstop.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, ensure_directories
from src.sigstop.data.align import align_price_data

def main():
    config = load_config()
    ensure_directories()

    symbols = config["data"]["symbols"]
    provider = config["data"]["provider"]

    symbol_paths = {}

    for symbol in symbols:
        path = RAW_DATA_DIR / provider / symbol / "daily.parquet"
        symbol_paths[symbol] = path

    aligned_df = align_price_data(symbol_paths = symbol_paths)

    output_path = PROCESSED_DATA_DIR / "aligned_prices.parquet"
    aligned_df.to_parquet(output_path, index = False)

    print("\nAligned data:")
    print(aligned_df.head())
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()