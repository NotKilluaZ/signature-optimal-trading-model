from __future__ import annotations
import numpy as np
import pandas as pd


VALID_ORIENTATIONS = (
    "GS_minus_beta_MS",
    "MS_minus_beta_GS",
)

def check_missing_cols(df: pd.DataFrame):
    required_cols = {"date", "MS", "GS"}
    missing = {col for col in required_cols if col not in df.columns}
    if missing:
        raise ValueError(f"Missing required columns for spread construction: {sorted(missing)}") 
    if df.empty:
        raise ValueError("Input pirce DataFrame is empty")

# Validate DataFrame inputs for aligned GS and MS data
def validate_spread_inputs(df: pd.DataFrame, beta: float, orientation: str) -> None:
    check_missing_cols(df)
    if not np.isfinite(beta):
        raise ValueError(f"Input Beta is not finite. Got: {beta}")
    if beta <= 0:
        raise ValueError(f"Beta must be greater than 0. Got: {beta}")
    if orientation not in VALID_ORIENTATIONS:
        raise ValueError(
            "Orientation is not one of the available options: "
            f"{', '.join(VALID_ORIENTATIONS)}"
        )
    if df[["MS", "GS"]].isna().any().any():
        raise ValueError("Input price columns contain NaN values")
    if (df[["MS", "GS"]] <= 0).any().any():
        raise ValueError("Input price columns contain negative values (must only be positive)")
    
# Takes in the aligned DataFrame, Beta value, and which of the two orientations
# Outputs a DataFrame with the columns: Data and Spread
def construct_spread(df: pd.DataFrame, beta: float, orientation: str) -> pd.DataFrame:
    validate_spread_inputs(df, beta, orientation)

    output = df[["date"]].copy()

    if orientation == "GS_minus_beta_MS":
        output["spread"] = df["GS"] - beta * df["MS"]
    elif orientation == "MS_minus_beta_GS":
        output["spread"] = df["MS"] - beta * df["GS"]
    else:
        raise ValueError(f"Invalid orientation: {orientation}")
    
    return output

# Normalizes each stock price by diving by inital price value
def construct_normalized_prices(df: pd.DataFrame) -> pd.DataFrame:
    check_missing_cols(df)
    if (df[["MS", "GS"]] <= 0).any().any():
        raise ValueError("Input price columns contain negative values (must only be positive)")
    
    output = df[["date"]].copy()
    output["MS_norm"] = df["MS"] / df["MS"].iloc[0]
    output["GS_norm"] = df["GS"] / df["GS"].iloc[0]

    return output
