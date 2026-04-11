import pandas as pd
from pathlib import Path

# Read parquet yfinance data
def load_symbol_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    return df

def align_price_data(symbol_paths: dict[str, Path], price_col: str = "adj_close") -> pd.DataFrame:
    dfs = []

    for symbol, path in symbol_paths.items():
        df = load_symbol_data(path)

        df = df[["date", price_col]].copy()
        df = df.rename(columns = {price_col: symbol})

        dfs.append(df)

    aligned_df = dfs[0]
    for df in dfs[1 : ]:
        aligned_df = aligned_df.merge(df, on = "date", how = "inner")

    aligned_df = aligned_df.sort_values("date").reset_index(drop = True)

    return aligned_df