"""Generate synthetic OHLCV data for freqtrade backtesting (offline / no exchange access)."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("user_data/data/binance")
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2025-02-01", tz="UTC")
TF = "5min"

pairs = {
    # pair filename stem -> (start_price, volatility per candle, drift per candle)
    "BTC_USDT-5m": (95000.0, 0.0025, 0.00003),
    "ETH_USDT-5m": (3300.0, 0.0035, 0.00002),
    "SOL_USDT-5m": (190.0, 0.0060, -0.00001),
}

for stem, (p0, vol, drift) in pairs.items():
    dates = pd.date_range(START, END, freq=TF, inclusive="left")
    n = len(dates)
    rng = np.random.default_rng(abs(hash(stem)) % (2**32))

    # random-walk close prices (geometric)
    rets = rng.normal(drift, vol, n)
    close = p0 * np.exp(np.cumsum(rets))

    open_ = np.empty(n)
    open_[0] = p0
    open_[1:] = close[:-1]

    # intrabar high/low around open & close
    hl_noise = np.abs(rng.normal(0, vol, n))
    high = np.maximum(open_, close) * (1 + hl_noise)
    low = np.minimum(open_, close) * (1 - hl_noise)

    volume = rng.uniform(5, 50, n) * (p0 / 100)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df["date"] = df["date"].astype("datetime64[ms, UTC]")

    path = OUT / f"{stem}.feather"
    df.reset_index(drop=True).to_feather(path, compression="lz4")
    print(f"wrote {path}  ({n} candles, {df['date'].min()} .. {df['date'].max()})")
