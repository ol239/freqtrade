import logging
from typing import Any

import numpy as np
import pandas as pd

from freqtrade.data.metrics import calculate_expectancy, calculate_max_drawdown
from freqtrade.optimize.optimize_reports.optimize_reports import generate_trading_stats
from freqtrade.util import fmt_coin, print_rich_table


logger = logging.getLogger(__name__)

# Thresholds used to turn raw statistics into coaching feedback.
MIN_TRADES_FOR_ANALYSIS = 20
OVERTRADING_TRADES_PER_DAY = 5
CONSECUTIVE_LOSS_WARNING = 4
CURRENT_LOSS_STREAK_WARNING = 3
STAKE_VARIATION_WARNING = 0.5
STOPLOSS_SHARE_WARNING = 0.5
LOW_PROFIT_FACTOR = 1.0
MARGINAL_PROFIT_FACTOR = 1.3
LOW_WINRATE = 0.4


def generate_trading_coach_report(trades: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze a trader's historic trades and build a "trader profile" of statistics
    that generate_coach_feedback() turns into coaching feedback.
    :param trades: Dataframe of trades, as returned by load_trades_from_db()
    :return: Dict of statistics describing the trader's behaviour
    """
    open_trade_count = int(trades["is_open"].sum()) if len(trades) else 0
    closed = trades.loc[~trades["is_open"]] if len(trades) else trades

    if len(closed) == 0:
        return {"trade_count": 0, "open_trade_count": open_trade_count}

    closed = closed.sort_values("close_date").reset_index(drop=True)

    stats = generate_trading_stats(closed)

    winning_profit = closed.loc[closed["profit_abs"] > 0, "profit_abs"].sum()
    losing_profit = closed.loc[closed["profit_abs"] < 0, "profit_abs"].sum()
    profit_factor = winning_profit / abs(losing_profit) if losing_profit else 0.0
    expectancy, expectancy_ratio = calculate_expectancy(closed)

    stake_mean = closed["stake_amount"].mean()
    stake_std = closed["stake_amount"].std() if len(closed) > 1 else 0.0
    stake_amount_variation = (stake_std / stake_mean) if stake_mean else 0.0

    trading_days = max(
        (closed["close_date"].max() - closed["open_date"].min()).total_seconds() / 86400, 1
    )
    trades_per_day = len(closed) / trading_days

    exit_reason_share = closed["exit_reason"].value_counts(normalize=True).to_dict()

    pair_profit = closed.groupby("pair")["profit_abs"].sum().sort_values()
    best_pair = (
        (str(pair_profit.index[-1]), float(pair_profit.iloc[-1])) if len(pair_profit) else None
    )
    worst_pair = (
        (str(pair_profit.index[0]), float(pair_profit.iloc[0])) if len(pair_profit) else None
    )

    # Length (and sign) of the streak of wins/losses immediately preceding the last trade.
    results = np.where(closed["profit_ratio"] > 0, 1, -1)
    last_sign = int(results[-1])
    streak_len = 0
    for r in results[::-1]:
        if r != last_sign:
            break
        streak_len += 1
    current_streak = streak_len * last_sign

    try:
        max_drawdown_abs = calculate_max_drawdown(closed).drawdown_abs
    except ValueError:
        max_drawdown_abs = 0.0

    return {
        **stats,
        "trade_count": len(closed),
        "open_trade_count": open_trade_count,
        "profit_total_abs": closed["profit_abs"].sum(),
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "expectancy_ratio": expectancy_ratio,
        "stake_amount_mean": stake_mean,
        "stake_amount_variation": stake_amount_variation,
        "trades_per_day": trades_per_day,
        "exit_reason_share": exit_reason_share,
        "best_pair": best_pair,
        "worst_pair": worst_pair,
        "current_streak": current_streak,
        "max_drawdown_abs": max_drawdown_abs,
    }


def generate_coach_feedback(stats: dict[str, Any]) -> list[str]:
    """
    Turn the statistics produced by generate_trading_coach_report() into a list of
    actionable, rule-based coaching messages.
    """
    trade_count = stats.get("trade_count", 0)
    if trade_count == 0:
        return ["No closed trades found yet - coaching feedback needs completed trades to analyze."]

    feedback: list[str] = []

    if trade_count < MIN_TRADES_FOR_ANALYSIS:
        feedback.append(
            f"Only {trade_count} closed trades available - the observations below are "
            f"indicative only. Aim for at least {MIN_TRADES_FOR_ANALYSIS} trades before "
            "drawing strong conclusions."
        )

    profit_factor = stats["profit_factor"]
    if profit_factor < LOW_PROFIT_FACTOR:
        feedback.append(
            f"Profit factor is {profit_factor:.2f} (below 1.0) - you are losing more than "
            "you win overall. Revisit entry criteria, tighten risk management, or reduce "
            "position size until this recovers."
        )
    elif profit_factor < MARGINAL_PROFIT_FACTOR:
        feedback.append(
            f"Profit factor is {profit_factor:.2f} - only marginally profitable, leaving "
            "little room for variance. Focus on cutting losing trades earlier or letting "
            "winners run further."
        )

    winrate = stats.get("winrate", 0.0)
    if winrate < LOW_WINRATE and stats["expectancy"] <= 0:
        feedback.append(
            f"Win-rate is {winrate:.0%} and expectancy per trade is not positive - your "
            "losers are not being offset by big enough winners. Reassess your reward:risk "
            "targets."
        )

    max_consecutive_losses = stats.get("max_consecutive_losses", 0)
    if max_consecutive_losses >= CONSECUTIVE_LOSS_WARNING:
        feedback.append(
            f"Longest losing streak was {max_consecutive_losses} trades in a row - consider "
            "a mandatory cooldown (e.g. stop for the day) after a few consecutive losses to "
            "avoid revenge trading."
        )

    current_streak = stats.get("current_streak", 0)
    if current_streak <= -CURRENT_LOSS_STREAK_WARNING:
        feedback.append(
            f"You are currently on a {abs(current_streak)}-trade losing streak - review "
            "your process before opening the next position rather than trying to "
            "'win it back'."
        )

    stake_amount_variation = stats.get("stake_amount_variation", 0.0)
    if stake_amount_variation > STAKE_VARIATION_WARNING:
        feedback.append(
            "Position sizes vary a lot between trades (stake coefficient of variation "
            f"{stake_amount_variation:.2f}) - inconsistent sizing makes risk hard to "
            "control. Consider risking a fixed percentage of your account per trade instead."
        )

    trades_per_day = stats.get("trades_per_day", 0.0)
    if trades_per_day > OVERTRADING_TRADES_PER_DAY:
        feedback.append(
            f"Averaging {trades_per_day:.1f} trades/day can be a sign of overtrading. Check "
            "that every entry still meets your strategy's rules rather than chasing activity."
        )

    exit_reason_share = stats.get("exit_reason_share", {})
    stoploss_share = sum(
        v
        for k, v in exit_reason_share.items()
        if "stop_loss" in k.lower() or "stoploss" in k.lower()
    )
    if stoploss_share > STOPLOSS_SHARE_WARNING:
        feedback.append(
            f"{stoploss_share:.0%} of trades are closed by stop-loss - consider whether "
            "entries are well-timed, or whether stops are placed too tight for the pair's "
            "volatility."
        )

    if not feedback:
        feedback.append(
            "No major red flags detected in this sample. Keep tracking your trades and re-run this "
            "coaching report regularly to confirm consistency over a larger sample size."
        )

    return feedback


def text_table_coach_report(
    stats: dict[str, Any], feedback: list[str], stake_currency: str
) -> None:
    """
    Print the trading-coach report (trader profile metrics + feedback) to the console.
    """
    if stats.get("trade_count", 0) == 0:
        print("No closed trades found - nothing to analyze yet.")
        return

    best_pair = stats.get("best_pair")
    worst_pair = stats.get("worst_pair")
    exit_reason_share = stats.get("exit_reason_share", {})
    exit_reasons = ", ".join(f"{k} {v:.0%}" for k, v in exit_reason_share.items())

    metrics = [
        ("Closed / Open trades", f"{stats['trade_count']} / {stats['open_trade_count']}"),
        ("Win / Draw / Loss", f"{stats['wins']} / {stats['draws']} / {stats['losses']}"),
        ("Win rate", f"{stats['winrate']:.2%}"),
        ("Total profit", fmt_coin(stats["profit_total_abs"], stake_currency)),
        ("Profit factor", f"{stats['profit_factor']:.2f}"),
        ("Expectancy (ratio)", f"{stats['expectancy']:.2f} ({stats['expectancy_ratio']:.2f})"),
        ("Max drawdown", fmt_coin(stats["max_drawdown_abs"], stake_currency)),
        ("Avg. stake amount", fmt_coin(stats["stake_amount_mean"], stake_currency)),
        ("Stake amount variation", f"{stats['stake_amount_variation']:.2f}"),
        ("Trades per day", f"{stats['trades_per_day']:.2f}"),
        (
            "Max consecutive wins / losses",
            f"{stats['max_consecutive_wins']} / {stats['max_consecutive_losses']}",
        ),
        ("Current streak", str(stats["current_streak"])),
        (
            "Best / Worst pair",
            f"{best_pair[0] if best_pair else 'N/A'} / {worst_pair[0] if worst_pair else 'N/A'}",
        ),
        ("Exit reasons", exit_reasons or "N/A"),
    ]
    print_rich_table(metrics, ["Metric", "Value"], summary="TRADER PROFILE", justify="left")

    print("\nCoaching feedback:")
    for line in feedback:
        print(f" - {line}")
