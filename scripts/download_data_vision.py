"""
Download Binance OHLCV data on a US-based CI runner without hitting the
geo-blocked api.binance.com (HTTP 451).

Two network calls normally reach api.binance.com and fail from a US IP:
  1. Loading markets (exchangeInfo) on exchange init.
  2. The REST "remaining data" fallback inside Binance.get_historic_ohlcv_fast.

We neutralise both: inject a synthetic markets map, and stub the base REST
OHLCV fetch to return empty. The bulk historical data is then served entirely
from https://data.binance.vision (a public archive that is NOT geo-blocked),
which freqtrade uses automatically for spot 1m/3m/5m timeframes.
"""

import sys

import ccxt
from pandas import DataFrame

from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS
from freqtrade.exchange.exchange import Exchange
from freqtrade.util.datetime_helpers import dt_ts


def _pairs_from_argv() -> list[str]:
    argv = sys.argv[1:]
    pairs: list[str] = []
    for i, a in enumerate(argv):
        if a in ("-p", "--pairs"):
            for tok in argv[i + 1 :]:
                if tok.startswith("-"):
                    break
                pairs.append(tok)
            break
    return pairs or ["BTC/USDT", "ETH/USDT"]


def _market(symbol: str) -> dict:
    base, quote = symbol.split("/")
    return {
        "id": base + quote,
        "symbol": symbol,
        "base": base,
        "quote": quote,
        "settle": None,
        "baseId": base,
        "quoteId": quote,
        "active": True,
        "type": "spot",
        "spot": True,
        "margin": False,
        "swap": False,
        "future": False,
        "option": False,
        "contract": False,
        "linear": None,
        "inverse": None,
        "contractSize": None,
        "taker": 0.001,
        "maker": 0.001,
        "percentage": True,
        "precision": {"amount": 8, "price": 2, "base": 8, "quote": 8},
        "limits": {
            "amount": {"min": 0.0, "max": None},
            "price": {"min": 0.0, "max": None},
            "cost": {"min": 5.0, "max": None},
            "leverage": {"min": None, "max": None},
        },
        "info": {},
    }


MARKETS = {p: _market(p) for p in _pairs_from_argv()}
TIMEFRAMES = {tf: tf for tf in ("1m", "3m", "5m", "15m", "1h", "4h", "1d")}


def _patched_reload_markets(self, force: bool = False, *, load_leverage_tiers: bool = True) -> None:
    for api in (self._api, self._api_async):
        api.markets = MARKETS
        api.markets_by_id = {m["id"]: [m] for m in MARKETS.values()}
        api.symbols = list(MARKETS)
        api.timeframes = TIMEFRAMES
        api.precisionMode = ccxt.DECIMAL_PLACES
    self._markets = MARKETS
    self._last_markets_refresh = dt_ts()


def _empty_ohlcv(self, *args, **kwargs) -> DataFrame:
    # Kill the REST fallback to api.binance.com; all data comes from the archive.
    return DataFrame(columns=DEFAULT_DATAFRAME_COLUMNS)


async def _empty_candle_history(self, pair, timeframe, candle_type, since_ms=None):
    # Binance.get_historic_ohlcv makes a REST "new pair listing date" probe
    # (since_ms=0) before choosing the vision path; return no candles so it
    # keeps the requested start date and never hits api.binance.com.
    return (pair, timeframe, candle_type, [])


Exchange.reload_markets = _patched_reload_markets
Exchange.validate_pairs = lambda self, pairs: None
Exchange._load_async_markets = lambda self, reload=False: None
# Binance.get_historic_ohlcv_fast calls super().get_historic_ohlcv (this base
# method) for the not-yet-archived tail; stub it so it never touches the REST API.
Exchange.get_historic_ohlcv = _empty_ohlcv
Exchange._async_get_candle_history = _empty_candle_history

from freqtrade.main import main  # noqa: E402


if __name__ == "__main__":
    main(sys.argv[1:])
