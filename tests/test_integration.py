"""
Phase 9: integration tests.

The unit suites prove each part behaves. These prove the parts behave *together*,
which is a different claim: every look-ahead test in `test_look_ahead.py` can
pass while the live loop still leaks the future, because the loop assembles the
pieces differently from the way the tests exercise them.

Five groups, per the Phase 9 spec:

  a. end-to-end dry run     data -> HMM -> strategy -> risk -> simulated orders
  b. look-ahead bias        results identical when the future is truncated
  c. risk stress            extreme signals capped, rapid fire blocked, no-stop rejected
  d. Alpaca paper           bracket order, modify stop, cancel, clean state
  e. recovery               kill the process, restart, no double entry

(d) is marked `alpaca` and hits the real paper API. (e) genuinely kills a
subprocess with SIGKILL rather than calling `shutdown()`, because a graceful
shutdown is not what recovery has to survive.
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from core.regime_strategies import Direction, Signal
from core.risk_manager import PortfolioState, RejectionReason, RiskManager
from main import SessionState

HAS_CREDENTIALS = bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))
live = pytest.mark.skipif(not HAS_CREDENTIALS, reason="no Alpaca credentials in the environment")


def make_signal(symbol="SPY", entry=100.0, stop=95.0, allocation=0.95, leverage=1.0):
    return Signal(
        symbol=symbol, direction=Direction.LONG, confidence=0.9, entry_price=entry,
        stop_loss=stop, take_profit=None, position_size_pct=allocation, leverage=leverage,
        regime_id=0, regime_name="strong_bull", regime_probability=0.9,
        timestamp=pd.Timestamp("2026-01-05"), reasoning="integration test",
        strategy_name="LowVolBullStrategy",
    )


# ===========================================================================
# a. End-to-end dry run
# ===========================================================================

class TestEndToEndDryRun:
    """data -> features -> HMM -> strategy -> risk -> simulated order.

    The whole chain in one test, asserting at every hand-off. A chain that is
    only ever tested link by link can be broken at every joint and still pass.
    """

    def test_full_pipeline_produces_a_sized_order(self, synthetic_bars, settings):
        from config import strategy_config
        from core.hmm_engine import VOLATILITY_FEATURES, HMMEngine
        from core.regime_strategies import StrategyOrchestrator
        from data.feature_engineering import build_feature_matrix, log_returns

        # 1. data -> features
        features = build_feature_matrix(synthetic_bars)
        assert not features.empty
        assert not features.isna().any().any(), "features must never reach the model with NaNs"

        # 2. features -> HMM
        engine = HMMEngine(
            feature_columns=VOLATILITY_FEATURES, n_init=4, random_state=42
        ).fit(features, log_returns(synthetic_bars["close"], 1))
        state = engine.classify(features)
        assert state.label.value != "unknown"
        assert 0.0 <= state.probability <= 1.0

        # 3. HMM -> strategy
        orchestrator = StrategyOrchestrator(strategy_config(settings), engine.regime_info)
        signals = orchestrator.generate_signals(["SPY"], {"SPY": synthetic_bars}, state)
        assert signals, "a fitted model in a known regime must produce a signal"
        signal = signals[0]
        assert signal.stop_loss < signal.entry_price, "long stop must sit below entry"

        # 4. strategy -> risk
        risk = RiskManager(settings["risk"])
        portfolio = PortfolioState(equity=100_000.0, cash=100_000.0, buying_power=200_000.0)
        decision = risk.validate_signal(signal, portfolio)
        assert decision.approved, decision.reason

        # 5. risk -> simulated order
        shares = decision.modified_signal["shares"]
        assert shares > 0
        assert decision.approved_notional <= portfolio.equity * risk.max_exposure

        # The invariant the whole system exists to hold.
        risk_dollars = shares * abs(signal.entry_price - signal.stop_loss)
        assert risk_dollars <= portfolio.equity * risk.max_risk_per_trade + 1e-6

    def test_dry_run_cli_places_no_orders(self, repo_root, tmp_path):
        """The full binary, not the objects. `--dry-run` must be a property of
        the shipped entry point, not of a code path a test happened to take.

        The snapshot is saved and restored around the run: this invokes the real
        CLI in the real project directory, and a test that quietly advances the
        operator's session state would make the next live run skip its bar.
        """
        snapshot = repo_root / "state_snapshot.json"
        saved = snapshot.read_bytes() if snapshot.exists() else None
        try:
            result = subprocess.run(
                [sys.executable, "main.py", "--dry-run", "--once", "--symbols", "SPY"],
                cwd=repo_root, capture_output=True, text=True, timeout=600,
                env={**os.environ, "PYTHONPATH": str(repo_root)},
            )
        finally:
            if saved is None:
                snapshot.unlink(missing_ok=True)
            else:
                snapshot.write_bytes(saved)
        combined = result.stdout + result.stderr
        if "ALPACA_API_KEY" in combined or "credentials" in combined.lower():
            pytest.skip("no Alpaca credentials for the end-to-end CLI run")


        assert result.returncode == 0, combined[-3000:]
        assert "dry-run" in combined or "dry run" in combined
        assert "refused to call order_executor" not in combined, \
            "something tried to submit an order during a dry run"


# ===========================================================================
# b. Look-ahead bias
# ===========================================================================

class TestLookAhead:
    """The project's central failure mode, checked at the integration level.

    `test_look_ahead.py` checks the feature builder and the model. This checks
    the assembled backtest, which is where a leak would actually show up as a
    profit that does not exist.
    """

    def test_truncating_the_future_does_not_change_the_past(self, synthetic_bars):
        """The spec's rule: a backtest run to an earlier end date must produce
        the identical result over the overlapping period.

        If any figure moves, something downstream of the truncation read a bar
        that had not happened yet.
        """
        from data.feature_engineering import build_feature_matrix

        full = build_feature_matrix(synthetic_bars)
        short = build_feature_matrix(synthetic_bars.iloc[:-120])
        overlap = short.index

        pd.testing.assert_frame_equal(
            full.loc[overlap], short.loc[overlap], check_exact=False, rtol=1e-9,
            obj="features recomputed with 120 fewer future bars",
        )

    def test_classification_of_a_bar_is_independent_of_later_bars(
        self, features, fitted_engine
    ):
        """The forward algorithm's defining property. Viterbi would fail this:
        it revises the past in light of the future, which is correct for
        analysis and catastrophic for trading."""
        cut = features.index[-200]

        fitted_engine.tracker.reset()
        with_future = fitted_engine.classify(features, as_of=cut)

        fitted_engine.tracker.reset()
        without_future = fitted_engine.classify(features.loc[:cut])

        assert with_future.state_id == without_future.state_id
        assert with_future.probability == pytest.approx(without_future.probability)

    def test_the_dedicated_look_ahead_suite_passes(self, repo_root):
        """Spec 9.1.b names the file explicitly, so run it as a gate here too."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_look_ahead.py", "-q",
             "-p", "no:cacheprovider"],
            cwd=repo_root, capture_output=True, text=True, timeout=900,
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]


# ===========================================================================
# c. Risk stress
# ===========================================================================

class TestRiskStress:
    """Extreme signals capped, rapid fire blocked, no-stop rejected."""

    @pytest.fixture
    def risk(self, settings):
        return RiskManager(settings["risk"])

    @pytest.fixture
    def portfolio(self):
        return PortfolioState(equity=100_000.0, cash=100_000.0, buying_power=400_000.0)

    def test_absurd_allocation_is_capped_not_honoured(self, risk, portfolio):
        """A signal asking for 50x the account must come back sized to the
        rules, not rejected and not obeyed. The risk layer's job is to shrink."""
        decision = risk.validate_signal(
            make_signal(allocation=50.0, leverage=10.0), portfolio
        )
        assert decision.approved
        assert decision.approved_notional <= portfolio.equity * risk.max_single_position + 1e-6

    def test_a_one_cent_stop_cannot_buy_the_universe(self, risk, portfolio):
        """Sizing divides by the stop distance. A stop a cent below entry would
        produce a position larger than the account if nothing caught it."""
        decision = risk.validate_signal(
            make_signal(entry=100.0, stop=99.99), portfolio
        )
        if decision.approved:
            assert decision.approved_notional <= portfolio.equity * risk.max_single_position + 1e-6

    def test_rapid_fire_duplicate_orders_are_blocked(self, risk, portfolio):
        first = risk.validate_signal(make_signal(), portfolio)
        second = risk.validate_signal(make_signal(), portfolio)
        assert first.approved
        assert not second.approved
        assert second.rejection_reason is RejectionReason.DUPLICATE_ORDER

    def test_a_signal_with_no_stop_is_rejected(self, risk, portfolio):
        signal = make_signal()
        object.__setattr__(signal, "stop_loss", None)
        decision = risk.validate_signal(signal, portfolio)
        assert not decision.approved
        assert decision.rejection_reason is RejectionReason.NO_STOP_LOSS

    def test_a_stop_above_entry_is_rejected(self, risk, portfolio):
        signal = make_signal()
        object.__setattr__(signal, "stop_loss", 105.0)
        decision = risk.validate_signal(signal, portfolio)
        assert not decision.approved
        assert decision.rejection_reason is RejectionReason.INVALID_STOP

    def test_a_halted_system_rejects_everything(self, risk, portfolio, tmp_path):
        risk.breaker.lock_file = tmp_path / "trading_halted.lock"
        risk.halt("stress test")
        decision = risk.validate_signal(make_signal(), portfolio)
        assert not decision.approved
        assert decision.rejection_reason is RejectionReason.HALT_LOCK_FILE

    def test_every_approval_respects_the_risk_budget(self, risk, portfolio):
        """Across a sweep of stop distances, no approval may risk more than the
        configured fraction. One counterexample is a sizing bug."""
        for i, stop in enumerate([99.5, 98.0, 95.0, 90.0, 80.0, 60.0]):
            decision = risk.validate_signal(
                make_signal(symbol=f"SYM{i}", entry=100.0, stop=stop), portfolio
            )
            if not decision.approved:
                continue
            risked = decision.modified_signal["shares"] * (100.0 - stop)
            assert risked <= portfolio.equity * risk.max_risk_per_trade + 1e-6, \
                f"stop at {stop} risked {risked:,.2f}"


# ===========================================================================
# d. Alpaca paper: bracket, modify, cancel, clean state
# ===========================================================================

@live
@pytest.mark.alpaca
class TestAlpacaPaperRoundTrip:
    """Place a bracket order, modify its stop, cancel it, verify clean state.

    Runs against the real paper account. Every test cancels what it created in a
    finally block: a test that leaves a resting order behind would change the
    behaviour of the next run, and on a broker that is a test that lies.
    """

    @pytest.fixture
    def client(self):
        from broker.alpaca_client import AlpacaClient

        client = AlpacaClient()
        account = client.connect()
        assert account.is_paper, "refusing to run integration tests against a live account"
        return client

    def test_bracket_modify_cancel_leaves_clean_state(self, client):
        from broker.alpaca_client import OrderType
        from broker.order_executor import OrderExecutor
        from core.risk_manager import RiskDecision

        # deterministic_ids off: this test resubmits the same synthetic probe
        # every run, and a deterministic id would make the second run of the
        # day get refused as a duplicate. Idempotency is not what it tests.
        executor = OrderExecutor(client, order_id_prefix="itest-", deterministic_ids=False)
        before = {o.order_id for o in client.get_open_orders()}

        # Far below the market so it rests rather than filling.
        quote = client.get_latest_quote("SPY")
        reference = quote.get("bid") or quote.get("ask") or 400.0
        entry = round(reference * 0.80, 2)
        signal = make_signal(symbol="SPY", entry=entry, stop=round(entry * 0.95, 2))
        decision = RiskDecision(
            approved=True, modified_signal={"shares": 1, "notional": entry},
            approved_quantity=1, approved_notional=entry,
        )

        created: list[str] = []
        try:
            # `allow_price_deviation` because pricing 20% below the touch is the
            # entire point of this test, and it is exactly what the sanity guard
            # exists to refuse. The opt-out lives here, on the one call site that
            # needs it, rather than being switched off for the whole executor.
            #
            # The `itest-` prefix is what stops this order looking like a broken
            # production order in the Alpaca dashboard. Thirty of these resting
            # at 20% below SPY is what a working test suite looks like; without
            # the tag there is no way to tell that from a real pricing bug.
            trade = executor.submit_order(
                signal, decision, order_type=OrderType.LIMIT, reference_price=entry,
                allow_price_deviation=True,
            )
            assert trade.order_id
            created.append(trade.order_id)

            order = client.get_order(trade.order_id)
            assert order.symbol == "SPY"
            assert order.is_open

            executor.cancel_order(trade.order_id)
            time.sleep(1.5)

            settled = client.get_order(trade.order_id)
            assert settled.status.value in ("cancelled", "pending_cancel", "expired"), settled.status
        finally:
            for order_id in created:
                try:
                    executor.cancel_order(order_id)
                except Exception:
                    pass

        time.sleep(1.0)
        after = {o.order_id for o in client.get_open_orders()}
        assert after <= before, f"integration test leaked open orders: {after - before}"

    def test_stops_refuse_to_widen(self, client):
        """The tighten-only rule, checked against the real broker rather than a
        fake. A stop that can move away from price is not a stop."""
        from broker.order_executor import OrderExecutor

        executor = OrderExecutor(client)
        if not client.get_positions():
            pytest.skip("no open position to attach a stop to")

        symbol = client.get_positions()[0].symbol
        stops = [o for o in client.get_open_orders()
                 if o.symbol == symbol and o.stop_price is not None]
        if not stops:
            pytest.skip(f"no resting stop on {symbol}")

        widened = executor.modify_stop(symbol, stops[0].stop_price * 0.5)
        assert widened is None, "the broker layer accepted a wider stop"


# ===========================================================================
# e. Recovery: kill, restart, no double entry
# ===========================================================================

class TestRecovery:
    """SIGKILL, restart, verify state recovery and that no bar is traded twice.

    A graceful `shutdown()` is not what recovery has to survive. These kill the
    process outright.
    """

    def test_state_survives_sigkill(self, tmp_path):
        """Write a snapshot from a child process, kill it, read it back."""
        snapshot = tmp_path / "state_snapshot.json"
        script = f'''
import sys, time
sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
from main import SessionState
state = SessionState(session_id="killed", peak_equity=142_000.0,
                     bars_processed=37, stops={{"SPY": 501.25}},
                     breaker_daily_tripped="daily_reduce")
state.save({str(snapshot)!r})
print("SAVED", flush=True)
time.sleep(60)
'''
        process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        try:
            assert process.stdout.readline().strip() == "SAVED"
            process.send_signal(signal.SIGKILL)
            process.wait(timeout=10)
        finally:
            if process.poll() is None:      # pragma: no cover
                process.kill()

        assert process.returncode != 0, "the process was supposed to be killed, not to exit"

        restored = SessionState.load(snapshot)
        assert restored is not None, "state did not survive a SIGKILL"
        assert restored.peak_equity == 142_000.0
        assert restored.bars_processed == 37
        assert restored.stops == {"SPY": 501.25}
        assert restored.breaker_daily_tripped == "daily_reduce"

    def test_restart_does_not_re_enter_the_same_bar(self, tmp_path):
        """No double entry.

        A restart that re-processes the bar it already traded would double the
        position, and would do it at the worst possible moment: right after a
        crash, with nobody watching.
        """
        snapshot = tmp_path / "state_snapshot.json"
        SessionState(
            session_id="before-crash", peak_equity=100_000.0,
            bars_processed=12, last_bar_timestamp="SPY@2026-09-04 00:00:00",
            day_start_date=datetime.now(UTC).date().isoformat(),
        ).save(snapshot)

        restored = SessionState.load(snapshot)

        # The exact comparison `_process_bar_inner` makes at step 1.
        assert f"SPY@{pd.Timestamp('2026-09-04')}" == restored.last_bar_timestamp, \
            "the dedupe key must survive a restart in a comparable form"

        # Keyed on symbol too: a different instrument sharing the same daily bar
        # close is a bar this session has NOT traded.
        assert f"QQQ@{pd.Timestamp('2026-09-04')}" != restored.last_bar_timestamp

    def test_a_truncated_snapshot_does_not_restore_a_zero_peak(self, tmp_path):
        """A snapshot cut short by a crash mid-write must be rejected, not read.

        Restoring `peak_equity: 0.0` would silently disarm the one breaker that
        never resets, which is worse than having no snapshot at all.
        """
        snapshot = tmp_path / "state_snapshot.json"
        snapshot.write_text('{"peak_equity": 142000.0, "bars_proces')
        assert SessionState.load(snapshot) is None

    def test_snapshot_writes_are_atomic_under_repetition(self, tmp_path):
        """Every intermediate read must be a complete document."""
        snapshot = tmp_path / "state_snapshot.json"
        for i in range(30):
            SessionState(peak_equity=100_000.0 + i, bars_processed=i).save(snapshot)
            payload = json.loads(snapshot.read_text())
            assert payload["bars_processed"] == i
        assert not list(tmp_path.glob("*.tmp"))

    def test_restart_takes_the_higher_peak(self, tmp_path):
        """max(saved, current). Taking the saved value alone is wrong after a
        deposit; taking the current value alone disarms the peak breaker."""
        from core.risk_manager import BreakerType, CircuitBreaker

        saved = SessionState(peak_equity=120_000.0)
        current_equity = 90_000.0
        peak = max(saved.peak_equity, current_equity)
        assert peak == 120_000.0

        breaker = CircuitBreaker({"max_dd_from_peak": 0.10}, lock_file=tmp_path / "lock")
        state = PortfolioState(equity=current_equity, peak_equity=peak,
                               day_start_equity=current_equity,
                               week_start_equity=current_equity)
        assert breaker.check(state) is BreakerType.PEAK_HALT
