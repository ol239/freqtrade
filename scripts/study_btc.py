"""
Study BTC/USDT price behaviour across years to inform strategy design.

Reads 5m OHLCV, resamples to daily, and reports:
  - Yearly returns and buy&hold behaviour
  - Market regimes (bull / bear / chop) via 200-day trend + slope
  - Volatility clustering (ATR%) and its autocorrelation
  - Trend persistence vs mean-reversion (autocorrelation of daily returns,
    and momentum vs reversal edge)
  - Drawdown profile
The point is to answer: does BTC trend or mean-revert, and when?
"""

import sys

import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "market_data/binance/BTC_USDT-5m.feather"


def load_daily(path: str) -> pd.DataFrame:
    df = pd.read_feather(path).set_index("date").sort_index()
    d = df.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return d


def main() -> None:
    d = load_daily(DATA)
    print(f"Loaded {len(d)} daily candles: {d.index.min().date()} -> {d.index.max().date()}")
    print(f"Price: {d['close'].iloc[0]:,.0f} -> {d['close'].iloc[-1]:,.0f}\n")

    d["ret"] = d["close"].pct_change()
    d["logret"] = np.log(d["close"]).diff()

    # ---- Yearly returns ----
    print("=== Yearly buy & hold returns ===")
    yearly = d["close"].resample("1YE").last().pct_change()
    first_year = d.index[0].year
    y0_ret = d[d.index.year == first_year]["close"].iloc[-1] / d["close"].iloc[0] - 1
    print(f"{first_year}: {y0_ret:+7.1%} (partial)")
    for ts, r in yearly.dropna().items():
        print(f"{ts.year}: {r:+7.1%}")

    # ---- Regime classification (200d trend) ----
    d["sma200"] = d["close"].rolling(200).mean()
    d["sma50"] = d["close"].rolling(50).mean()
    d["above200"] = d["close"] > d["sma200"]
    frac_above = d["above200"].mean()
    print(f"\n=== Regime ===")
    print(f"Days price above 200d SMA: {frac_above:.1%}")
    # forward 20d return conditioned on regime
    d["fwd20"] = d["close"].shift(-20) / d["close"] - 1
    up = d.loc[d["above200"], "fwd20"].mean()
    dn = d.loc[~d["above200"], "fwd20"].mean()
    print(f"Avg next-20d return when ABOVE 200d SMA: {up:+.2%}")
    print(f"Avg next-20d return when BELOW 200d SMA: {dn:+.2%}")

    # ---- Volatility clustering ----
    d["atr%"] = (d["high"] - d["low"]) / d["close"]
    ac_vol = d["atr%"].autocorr(1)
    print(f"\n=== Volatility ===")
    print(f"Mean daily range (ATR%): {d['atr%'].mean():.2%}")
    print(f"Volatility autocorrelation (lag 1d): {ac_vol:.2f}  (high => vol clusters)")

    # ---- Trend vs mean-reversion ----
    ac1 = d["logret"].autocorr(1)
    ac5 = d["logret"].rolling(5).sum().autocorr(5)
    print(f"\n=== Trend vs mean-reversion (daily) ===")
    print(f"Autocorr of daily returns (lag 1): {ac1:+.3f}")
    print(f"Autocorr of 5d returns (lag 5):    {ac5:+.3f}  (>0 => momentum, <0 => reversal)")
    # momentum edge: does a positive 20d momentum predict positive next 20d?
    d["mom20"] = d["close"] / d["close"].shift(20) - 1
    edge_mom = d.loc[d["mom20"] > 0, "fwd20"].mean() - d.loc[d["mom20"] <= 0, "fwd20"].mean()
    print(f"Momentum edge (next-20d when 20d-mom>0 minus <=0): {edge_mom:+.2%}")

    # ---- Drawdown ----
    roll_max = d["close"].cummax()
    dd = d["close"] / roll_max - 1
    print(f"\n=== Drawdown (buy & hold) ===")
    print(f"Max drawdown: {dd.min():.1%}")
    print(f"Time underwater (>10% dd): {(dd < -0.10).mean():.1%} of days")


if __name__ == "__main__":
    main()
