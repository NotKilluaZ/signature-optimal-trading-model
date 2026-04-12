import concurrent.futures
import pandas as pd
import yfinance as yf
from src.sigstop.data.schema import prepare_and_validate_schema

_DOWNLOAD_TIMEOUT_SECONDS = 60

def download_yfinance_symbol(symbol: str, start_date: str, end_date: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
    def _fetch() -> pd.DataFrame:
        return yf.download(
            symbol,
            start = start_date,
            end = end_date,
            interval = interval,
            auto_adjust = auto_adjust,
            progress = False,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers = 1) as executor:
        future = executor.submit(_fetch)
        try:
            df = future.result(timeout = _DOWNLOAD_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"yfinance download for {symbol!r} timed out after {_DOWNLOAD_TIMEOUT_SECONDS}s. "
                "Check your network connection or Yahoo Finance availability."
            )

    if df.empty:
        raise ValueError(f"No data returned for symbol: {symbol}")
    
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