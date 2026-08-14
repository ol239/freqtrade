"""
Run a freqtrade backtest fully OFFLINE (no exchange/network access).

This environment's egress policy blocks crypto-exchange APIs, so freqtrade
cannot fetch market metadata on startup. We inject a small synthetic markets
map for our whitelisted pairs and neutralise the network calls, then hand off
to the normal freqtrade CLI. The backtest maths run exactly as usual on the
local (synthetic) OHLCV data.
"""

import json
import sys

import ccxt

from freqtrade.exchange.exchange import Exchange
from freqtrade.util.datetime_helpers import dt_ts


def _pairs_from_argv() -> list[str]:
    """Read the exchange pair_whitelist from the --config file passed on the CLI."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a in ("-c", "--config") and i + 1 < len(argv):
            try:
                with open(argv[i + 1]) as fh:
                    cfg = json.load(fh)
                pairs = cfg.get("exchange", {}).get("pair_whitelist")
                if pairs:
                    return pairs
            except Exception:
                pass
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


PAIRS = _pairs_from_argv()
TIMEFRAMES = {tf: tf for tf in ("1m", "5m", "15m", "1h", "4h", "1d")}


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


MARKETS = {p: _market(p) for p in PAIRS}


def _patched_reload_markets(self, force: bool = False, *, load_leverage_tiers: bool = True) -> None:
    for api in (self._api, self._api_async):
        api.markets = MARKETS
        api.markets_by_id = {m["id"]: [m] for m in MARKETS.values()}
        api.symbols = list(MARKETS)
        api.timeframes = TIMEFRAMES
        api.precisionMode = ccxt.DECIMAL_PLACES
    self._markets = MARKETS
    self._last_markets_refresh = dt_ts()


# Neutralise network-backed calls that would otherwise reach the exchange.
Exchange.reload_markets = _patched_reload_markets
Exchange.validate_pairs = lambda self, pairs: None
Exchange._load_async_markets = lambda self, reload=False: None

from freqtrade.main import main  # noqa: E402


if __name__ == "__main__":
    main(sys.argv[1:])
