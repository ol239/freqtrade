"""Resample the 5m BTC feather into 1h and 1d feather files for freqtrade."""

from pathlib import Path

import pandas as pd

SRC = Path("market_data/binance/BTC_USDT-5m.feather")
OUTDIR = SRC.parent

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

df = pd.read_feather(SRC).set_index("date").sort_index()

for tf, rule in [("1h", "1h"), ("1d", "1D")]:
    r = df.resample(rule).agg(AGG).dropna().reset_index()
    out = OUTDIR / f"BTC_USDT-{tf}.feather"
    r.to_feather(out, compression="lz4")
    print(f"wrote {out}: {len(r)} candles  {r['date'].min()} -> {r['date'].max()}")
