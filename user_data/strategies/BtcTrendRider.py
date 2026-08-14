import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    IStrategy,
    stoploss_from_absolute,
)


class BtcTrendRider(IStrategy):
    """
    Daily trend-following BTC strategy, designed from 6 years of price behaviour.

    Findings that shaped it:
      - BTC has a positive medium-term momentum edge -> trade WITH the trend.
      - Buy & hold suffers a -76% drawdown (2022 bear) -> exit to cash when the
        trend breaks; the main value-add is drawdown reduction.
      - 1h trading gets whipsawed to death -> trade the DAILY timeframe: fewer,
        larger positions that ride the big moves.
      - Volatility clusters -> a wide ATR chandelier stop is the crash backstop,
        while a moving-average exit handles normal trend weakening.

    Long-only (spot), trades the 1d timeframe.
    """

    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "1d"
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 220

    minimal_roi = {"0": 100}  # disabled: let trends run
    stoploss = -0.50  # hard safety net; custom_stoploss (ATR) is the real stop
    trailing_stop = False
    use_custom_stoploss = True

    # Hyperoptable parameters
    atr_mult = DecimalParameter(2.0, 12.0, default=4.0, decimals=1, space="sell", optimize=True)
    fast_ma = IntParameter(20, 80, default=50, space="buy", optimize=True)
    trend_ma = IntParameter(100, 200, default=200, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["sma_fast"] = ta.SMA(dataframe, timeperiod=self.fast_ma.value)
        dataframe["sma_trend"] = ta.SMA(dataframe, timeperiod=self.trend_ma.value)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] > dataframe["sma_trend"])  # long-term uptrend / bull regime
            & (dataframe["sma_fast"] > dataframe["sma_trend"])  # fast MA above trend MA
            & (dataframe["close"] > dataframe["ema20"])  # short-term momentum up
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["sma_trend"]),  # regime break -> to cash
            "exit_long",
        ] = 1
        return dataframe

    def custom_stoploss(
        self, pair, trade, current_time, current_rate, current_profit, **kwargs
    ) -> float | None:
        # Chandelier exit: trail the stop at (highest price since entry) - k*ATR.
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return None
        atr = df["atr"].iloc[-1]
        if pd.isna(atr) or atr <= 0:
            return None
        sl_price = trade.max_rate - self.atr_mult.value * atr
        return stoploss_from_absolute(sl_price, current_rate, is_short=trade.is_short)
