from __future__ import annotations
import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "symbol",
    "source",
]

# Turn all dataframe column names into lowercase
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.columns = [col.lower() for col in df.columns]
    return df

# Ensure DataFrame contains all required columns
def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) from DataFrame: {missing}")
    
# Ensure dates in DataFrame are in datetime format
def enforce_date_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors = "raise")
    return df


# Return DataFrame sorted in ascending datetime
def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df.sort_values(by = "date", ascending = True).reset_index(drop = True)

# Ensure no duplicate dates per stock symbol
def validate_no_duplicate_dates(df: pd.DataFrame) -> None:
    duplicate_mask = df.duplicated(subset = ["symbol", "date"], keep = False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, ["symbol", "date"]]
        raise ValueError(
            "Duplicate dates found.\n"
            f"Duplicate rows:\n{duplicates.to_string(index=False)}"
        )
    
# Raise ValueError if any columns in DataFrame are not lowercase
def validate_columns_lowercase(df: pd.DataFrame) -> None:
    non_lowercase = [col for col in df.columns if col != col.lower()]
    if non_lowercase:
        raise ValueError(f"Column name(s) not lowercase found: {non_lowercase}")
    
# Validate Schema
def validate_schema(df: pd.DataFrame) -> None:
    validate_columns_lowercase(df)
    validate_no_duplicate_dates(df)
    validate_no_duplicate_dates(df)

# Validate and Prepare DataFrame to fit expected formatting
def prepare_and_validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rules being checked:
    - all column names lowercase
    - date must be datetime
    - rows sorted by date ascending
    - no duplicate dates per symbol
    """

    df = standardize_column_names(df)
    validate_required_columns(df)
    df = enforce_date_datetime(df)
    df = sort_by_date(df)
    validate_schema(df)
    return df