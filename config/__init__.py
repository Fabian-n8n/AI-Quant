"""
Settings and credential loading.

Small addition to the tutorial's structure, which shows config/ holding only
the two YAML files. Something has to read them, and putting the loader here
keeps every other module free of file paths and yaml imports.

Implemented in Phase 7, when the orchestrator became the first caller that
needs the whole config validated up front rather than one section at a time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent
ROOT_DIR = CONFIG_DIR.parent

SETTINGS_PATH = CONFIG_DIR / "settings.yaml"
CREDENTIALS_PATH = CONFIG_DIR / "credentials.yaml"
ENV_PATH = ROOT_DIR / ".env"

# Sections that must exist. A config missing one of these is not "partially
# configured", it is a system with a silently disabled subsystem.
REQUIRED_SECTIONS = ("broker", "hmm", "strategy", "risk", "backtest", "monitoring")

# Keys within a section that have no safe default. Anything not listed here may
# fall back to the value baked into the class that reads it; anything listed
# here must be present, because guessing it would change what the system does.
REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "broker": ("paper_trading", "symbols", "timeframe"),
    "hmm": ("min_train_bars", "stability_bars", "min_confidence", "zscore_lookback"),
    "strategy": ("rebalance_threshold",),
    "risk": (
        "max_risk_per_trade",
        "max_exposure",
        "max_leverage",
        "daily_dd_halt",
        "weekly_dd_halt",
        "max_dd_from_peak",
    ),
    "backtest": ("initial_capital", "train_window", "test_window"),
    "monitoring": ("dashboard_refresh_seconds", "alert_rate_limit_minutes"),
}


class ConfigError(RuntimeError):
    """Raised when the config is unusable. Never downgraded to a warning."""


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Read settings.yaml and return it as a nested dict.

    Validates that every expected section is present rather than returning
    partial config. A missing 'risk' block must fail loudly at startup, not
    silently disable the circuit breakers at 3am.
    """
    import yaml

    target = Path(path) if path else SETTINGS_PATH
    if not target.exists():
        raise ConfigError(f"settings file not found: {target}")

    with open(target) as fh:
        settings = yaml.safe_load(fh)

    if not isinstance(settings, dict):
        raise ConfigError(f"{target} did not parse to a mapping")

    missing = [s for s in REQUIRED_SECTIONS if s not in settings]
    if missing:
        raise ConfigError(f"{target} is missing required sections: {', '.join(missing)}")

    for section, keys in REQUIRED_KEYS.items():
        absent = [k for k in keys if settings.get(section, {}).get(k) is None]
        if absent:
            raise ConfigError(
                f"{target} section '{section}' is missing required keys: {', '.join(absent)}"
            )

    if not settings["broker"]["symbols"]:
        raise ConfigError("broker.symbols is empty: there is nothing to trade")

    return settings


def strategy_config(settings: dict[str, Any]) -> dict[str, Any]:
    """The strategy block plus the two values it borrows from other sections.

    `min_confidence` lives under hmm and `max_leverage` under risk, but the
    orchestrator needs both. Assembling that here means the three callers that
    build an orchestrator cannot drift apart on which keys they copy across.
    """
    config = dict(settings["strategy"])
    config["min_confidence"] = settings["hmm"]["min_confidence"]
    config["max_leverage"] = settings["risk"]["max_leverage"]
    return config


def load_credentials() -> dict[str, Any]:
    """Read Alpaca credentials, environment first, credentials.yaml as fallback.

    Raises if any are missing rather than falling back to a default, because a
    silently missing key is how you end up pointing at the wrong account.

    Returns the values so a caller can pass them explicitly. It never logs them,
    and callers should not either: the only safe things to print about a
    credential are its length and its first two characters.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_PATH)
    except ImportError:
        pass

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    paper_env = os.getenv("ALPACA_PAPER")

    if (not api_key or not secret_key) and CREDENTIALS_PATH.exists():
        import yaml

        with open(CREDENTIALS_PATH) as fh:
            raw = yaml.safe_load(fh) or {}
        alpaca = raw.get("alpaca", raw)
        api_key = api_key or alpaca.get("api_key")
        secret_key = secret_key or alpaca.get("secret_key")
        if paper_env is None and "paper" in alpaca:
            paper_env = str(alpaca["paper"])

    if not api_key or not secret_key:
        raise ConfigError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY not found in the environment, "
            f"{ENV_PATH.name} or {CREDENTIALS_PATH.name}. Copy .env.example to .env "
            "and fill them in. Never paste keys into a chat window."
        )

    paper = str(paper_env if paper_env is not None else "true").strip().lower() not in (
        "false",
        "0",
        "no",
    )
    return {"api_key": api_key, "secret_key": secret_key, "paper": paper}


def assert_paper_mode(settings: dict[str, Any]) -> None:
    """Hard guard. Raises unless both settings.yaml and the environment agree
    this is a paper account.

    Called at startup by the orchestrator. Removing this call is a deliberate,
    reviewed act, not something that happens during a refactor.

    Three independent sources have to agree: settings.yaml, ALPACA_PAPER, and
    the key prefix Alpaca issues (PK for paper, AK for live). Any one of them
    dissenting stops startup. Agreement between two while the third disagrees is
    the interesting case, and it is exactly the case that means someone edited
    one file and forgot another.
    """
    if not settings.get("broker", {}).get("paper_trading", True):
        raise ConfigError(
            "broker.paper_trading is false in settings.yaml. Live trading is a "
            "deliberate act: read the README FAQ on live trading, then run with --i-understand-live."
        )

    credentials = load_credentials()
    if not credentials["paper"]:
        raise ConfigError(
            "ALPACA_PAPER is false but settings.yaml says paper_trading: true. "
            "Refusing to start with the two disagreeing about which account this is."
        )

    prefix = credentials["api_key"][:2].upper()
    if prefix == "AK":
        raise ConfigError(
            "ALPACA_API_KEY looks like a LIVE key (AK...) but paper mode was "
            "requested. Refusing to start."
        )
