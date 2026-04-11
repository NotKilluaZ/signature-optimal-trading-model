import pandas as pd

def create_formation_trading_split(df: pd.DataFrame, formation_days = 252, trading_days = 252):
    total_required = formation_days + trading_days

    if len(df) < total_required:
        raise ValueError(
            f"Not enough days-worth of data: need {total_required}, only got {len(df)}"
        )
    
    formation = df.iloc[ : formation_days].copy()
    trading = df.iloc[formation_days : total_required].copy()

    metadata = {
        "formation_start": str(formation["date"].min()),
        "formation_end": str(formation["date"].max()),
        "trading_start": str(trading["date"].min()),
        "trading_end": str(trading["date"].max()),
        "formation_days": formation_days,
        "trading_days": trading_days,
    }

    return formation, trading, metadata