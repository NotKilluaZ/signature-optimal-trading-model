import pandas as pd
import yfinance as yf
from src.sigstop.data.schema import prepare_and_validate_schema

def download_yfinance_symbol(symbol: str, start_date: str, end_date: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start = start_date,
        end = end_date,
        interval = interval,
        auto_adjust = auto_adjust,
        progress = False,
    )

    if df.empty:
        raise ValueError(f"No data return for symbol: {symbol}")
    
    # Reset DataFrame to be sequential default indexing
    df = df.reset_index()
    df = df.rename(columns = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })
    df["symbol"] = symbol
    df["source"] = "yfinance"

    df = prepare_and_validate_schema(df)
    
    return df