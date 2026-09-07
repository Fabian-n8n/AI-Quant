#!/usr/bin/env python3
"""
The gate between paper trading and real money.

Phase 10.

Six checks. All six must pass before `main.py` will run in live mode. Paper
mode is unaffected and always has been: the point is not to make the system
hard to use, it is to make ONE transition hard, the one that is irreversible.

WHY THIS EXISTS AS A SCRIPT AND NOT A CHECKLIST
-----------------------------------------------
A checklist in a README is a list of things you intended to verify. Every item
here was already written down in docs/, unticked, and would have stayed unticked
while the system happily accepted `--i-understand-live`. The difference between
a document and a gate is that the gate is checked by the machine at the moment
it matters, not by a person at the moment they are keen.

WHAT IT DOES NOT CHECK
----------------------
Whether the strategy is good. It checks whether there is *evidence*, which is a
much weaker claim and still one this system cannot currently make. Expect a
FAIL. That is the honest result today: the strategy loses to buy-and-hold
out-of-sample and has zero closed paper trades. A preflight that passed would
be lying.

    python scripts/preflight.py            # human readable, exit 0 or 1
    python scripts/preflight.py --json     # for CI
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MIN_CLOSED_TRADES = 30
MAX_BACKTEST_AGE_DAYS = 30


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    #: What to do about it. A failure you cannot act on is just bad news.
    remedy: str = ""
    blocking: bool = True


@dataclass
class Preflight:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.blocking)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail,
                 "remedy": c.remedy, "blocking": c.blocking}
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def check_closed_trades(repo) -> Check:
    """Thirty closed paper trades before real money.

    Thirty is not a magic number, it is roughly where a win rate stops being
    indistinguishable from a coin flip. Below it, "it works" and "it has been
    lucky" produce the same equity curve.
    """
    n = repo.closed_trade_count() if repo else 0
    return Check(
        name=f"At least {MIN_CLOSED_TRADES} closed paper trades",
        passed=n >= MIN_CLOSED_TRADES,
        detail=f"{n} closed",
        remedy=("Keep paper trading. At one daily decision this is months, "
                "which is the point rather than an inconvenience."),
    )


def check_expectancy(repo) -> Check:
    """Positive average P&L per closed trade.

    Reported with the win rate, because expectancy alone hides its own shape:
    one outsized winner among nineteen losers is a positive number and not a
    strategy.
    """
    stats = repo.expectancy() if repo else {"trades": 0, "expectancy": 0.0, "win_rate": 0.0}
    if not stats["trades"]:
        return Check(
            name="Positive expectancy per trade",
            passed=False,
            detail="no closed trades to compute it from",
            remedy="Same as above: this needs a trading history to exist.",
        )
    return Check(
        name="Positive expectancy per trade",
        passed=stats["expectancy"] > 0,
        detail=(f"${stats['expectancy']:+,.2f} per trade over {stats['trades']} "
                f"trades, {stats['win_rate']:.0%} win rate"),
        remedy="A negative expectancy is the strategy telling you it does not work.",
    )


def check_backtest_freshness() -> Check:
    """Walk-forward results exist and are recent.

    Age matters because a backtest is a claim about a market that has since
    moved on. A result from last year describes a regime that may no longer
    exist, and using it to justify going live today is using evidence about a
    different thing.
    """
    results = ROOT / "backtest" / "results"
    comparisons = sorted(results.glob("*/benchmark_comparison.csv"))
    if not comparisons:
        return Check(
            name=f"Walk-forward results under {MAX_BACKTEST_AGE_DAYS} days old",
            passed=False,
            detail="no benchmark comparison has been exported",
            remedy="python main.py --backtest --compare --export",
        )

    newest = max(comparisons, key=lambda p: p.stat().st_mtime)
    age = (datetime.now().timestamp() - newest.stat().st_mtime) / 86400
    return Check(
        name=f"Walk-forward results under {MAX_BACKTEST_AGE_DAYS} days old",
        passed=age <= MAX_BACKTEST_AGE_DAYS,
        detail=f"{newest.parent.name}: {age:.0f} days old",
        remedy="python main.py --backtest --compare --export",
    )


def check_beats_benchmarks() -> Check:
    """Beats buy-and-hold, 200-day SMA trend, and random entry.

    All three, on total return, out-of-sample. Random entry is the one that
    matters most and is usually left out: a strategy that cannot beat coin
    flips with the same exposure has not found anything, it has just been in
    the market during an up year.
    """
    import csv

    results = ROOT / "backtest" / "results"
    comparisons = sorted(results.glob("*/benchmark_comparison.csv"))
    if not comparisons:
        return Check(
            name="Beats buy-and-hold, SMA trend, and random entry",
            passed=False,
            detail="no comparison to read",
            remedy="python main.py --backtest --compare --export",
        )

    newest = max(comparisons, key=lambda p: p.stat().st_mtime)
    with open(newest) as fh:
        rows = {r["strategy"]: r for r in csv.DictReader(fh)}

    def total(name: str) -> float | None:
        row = rows.get(name)
        if not row or not row.get("total_return"):
            return None
        try:
            return float(row["total_return"])
        except ValueError:
            return None

    own = total("regime-trader")
    if own is None:
        return Check(
            name="Beats buy-and-hold, SMA trend, and random entry",
            passed=False, detail="strategy row missing from the comparison",
            remedy="Re-export the backtest.",
        )

    beaten, lost_to = [], []
    for label, key in (("buy-and-hold", "buy_and_hold"),
                       ("SMA-200 trend", "sma_200_trend"),
                       ("random entry", "random_allocation_100_seeds")):
        benchmark = total(key)
        if benchmark is None:
            lost_to.append(f"{label} (not measured)")
        elif own > benchmark:
            beaten.append(label)
        else:
            lost_to.append(f"{label} ({benchmark:+.1%} vs {own:+.1%})")

    return Check(
        name="Beats buy-and-hold, SMA trend, and random entry",
        passed=not lost_to,
        detail=(f"strategy {own:+.1%}. Loses to: {', '.join(lost_to)}"
                if lost_to else f"beats all three, {own:+.1%}"),
        remedy=("Losing to random entry means the regime signal is adding "
                "nothing. That is a strategy problem, not a settings problem."),
    )


def check_every_position_has_a_stop() -> Check:
    """Every open position has a live stop order at the broker.

    Not a stop_loss float in a config file. A resting order, at the broker,
    that works with this process dead.
    """
    try:
        from broker.alpaca_client import AlpacaClient, OrderType

        client = AlpacaClient()
        client.connect()
        positions = client.get_positions()
        if not positions:
            return Check(name="Every open position has a live stop", passed=True,
                         detail="no open positions")

        protected = {
            o.symbol for o in client.get_open_orders()
            if o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT,
                                OrderType.TRAILING_STOP)
        }
        naked = sorted({p.symbol for p in positions} - protected)
        return Check(
            name="Every open position has a live stop",
            passed=not naked,
            detail=(f"{', '.join(naked)} unprotected" if naked
                    else f"all {len(positions)} protected"),
            remedy="Start the engine once: startup reconciliation places missing stops.",
        )
    except Exception as exc:
        return Check(
            name="Every open position has a live stop",
            passed=False,
            detail=f"could not reach the broker: {exc}",
            remedy="Check ALPACA_API_KEY and ALPACA_SECRET_KEY.",
        )


def check_not_halted(repo) -> Check:
    """No breaker tripped and no halt lock on disk.

    The lock requires manual deletion by design. Going live while one exists
    would mean the first thing real money does is ignore a stop signal the
    system raised for itself.
    """
    lock = ROOT / "trading_halted.lock"
    active = repo.active_breakers() if repo else []

    problems = []
    if lock.exists():
        problems.append(f"{lock.name} present")
    if active:
        problems.append(f"breaker tripped: {', '.join(active)}")

    return Check(
        name="No breaker tripped, no halt lock",
        passed=not problems,
        detail="; ".join(problems) if problems else "clear",
        remedy=("Read the lock file before deleting it. It says why the system "
                "stopped, and deleting it unread is the habit this is designed "
                "to prevent."),
    )


# ---------------------------------------------------------------------------

def run_preflight(db_path: Path | None = None) -> Preflight:
    """Run every check. Never raises: a crashed preflight must read as a fail."""
    from data.repository import DEFAULT_DB, Repository

    result = Preflight()
    repo = None
    try:
        repo = Repository(db_path or DEFAULT_DB)
        repo.migrate()
    except Exception as exc:
        result.add(Check(name="State database readable", passed=False,
                         detail=str(exc), remedy="Check data/state.db."))
        repo = None

    for check in (
        check_closed_trades(repo),
        check_expectancy(repo),
        check_backtest_freshness(),
        check_beats_benchmarks(),
        check_every_position_has_a_stop(),
        check_not_halted(repo),
    ):
        result.add(check)

    if repo is not None:
        repo.close()
    return result


def render(result: Preflight) -> str:
    lines = ["", "Preflight: live trading readiness", "=" * 58, ""]
    for check in result.checks:
        mark = "PASS" if check.passed else "FAIL"
        lines.append(f"  [{mark}]  {check.name}")
        lines.append(f"          {check.detail}")
        if not check.passed and check.remedy:
            lines.append(f"          -> {check.remedy}")
        lines.append("")

    failed = [c for c in result.checks if not c.passed]
    lines.append("=" * 58)
    if result.passed:
        lines += [
            "  READY. All checks pass.",
            "",
            "  This says there is evidence, not that the strategy is good.",
            "  Size the first live position as though it will lose.",
        ]
    else:
        lines += [
            f"  NOT READY. {len(failed)} of {len(result.checks)} checks failed.",
            "",
            "  main.py will refuse --i-understand-live until these pass.",
            "  Paper trading is unaffected.",
        ]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--db", type=Path, help="state database to read")
    args = parser.parse_args(argv)

    result = run_preflight(args.db)
    print(json.dumps(result.as_dict(), indent=2) if args.json else render(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
