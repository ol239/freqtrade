import logging
from typing import Any


logger = logging.getLogger(__name__)


def start_trading_coach(args: dict[str, Any]) -> None:
    """
    Analyze the trades stored in the database and print a trader-profile report with
    rule-based coaching feedback.
    """
    import json

    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.data.btanalysis import load_trades_from_db
    from freqtrade.enums import RunMode
    from freqtrade.exceptions import ConfigurationError
    from freqtrade.misc import parse_db_uri_for_logging
    from freqtrade.optimize.optimize_reports import (
        generate_coach_feedback,
        generate_trading_coach_report,
        text_table_coach_report,
    )

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    if "db_url" not in config:
        raise ConfigurationError("--db-url is required for this command.")

    logger.info(f'Using DB: "{parse_db_uri_for_logging(config["db_url"])}"')
    trades = load_trades_from_db(config["db_url"])

    stats = generate_trading_coach_report(trades)
    feedback = generate_coach_feedback(stats)

    if config.get("print_json", False):
        print(json.dumps({"stats": stats, "feedback": feedback}, indent=4, default=str))
    else:
        text_table_coach_report(stats, feedback, config.get("stake_currency", "USDT"))
