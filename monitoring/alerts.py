"""
Alerts: console, log file, email and webhook.

Alert on things that need a human. Not on things that are merely interesting:
an alert that fires ten times a day gets muted within a week, and then the one
that mattered gets muted with it.

## The seven triggers

Per the Phase 8 spec, with the severity each is actually worth:

| Trigger | Level | Why that level |
|---|---|---|
| Circuit breaker | CRITICAL | needs manual intervention to clear |
| Data feed down | CRITICAL | the system is trading blind or not at all |
| API lost | CRITICAL | orders and stops may not be reaching the broker |
| Large P&L move | WARNING | worth knowing today, not worth waking up for |
| Flicker exceeded | WARNING | the model is unsure; size is already cut |
| Regime change | INFO | routine. It is the system working, not failing |
| HMM retrained | INFO | routine, weekly |

**INFO alerts go to console and log only.** They never send email or hit a
webhook, because a regime change is the system doing its job and an inbox that
receives one every few days is an inbox that stops being read. `alert_min_level`
in settings controls the threshold if you disagree.

## Rate limiting

One per event type per 15 minutes, per spec. Keyed on the trigger name rather
than the message, so "breaker daily_reduce at 2.1%" and "breaker daily_reduce at
2.4%" collapse into one. Keying on the message would let a value that changes
every bar defeat the limit entirely.

Nothing here raises. A delivery failure is logged and swallowed: an alert about
a circuit breaker must not become a second exception that takes down the loop
that was trying to report the first one.
"""

from __future__ import annotations

import logging
import os
import smtplib
import threading
import time
from email.message import EmailMessage
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


_LEVEL_ORDER = {AlertLevel.INFO: 0, AlertLevel.WARNING: 1, AlertLevel.CRITICAL: 2}


class AlertTrigger(str, Enum):
    """The spec's seven. Also the rate-limit key, so one noisy condition
    cannot crowd out a different one."""
    REGIME_CHANGE = "regime_change"
    CIRCUIT_BREAKER = "circuit_breaker"
    LARGE_PNL = "large_pnl"
    DATA_FEED_DOWN = "data_feed_down"
    API_LOST = "api_lost"
    HMM_RETRAINED = "hmm_retrained"
    FLICKER_EXCEEDED = "flicker_exceeded"
    # Beyond the spec, but each is a condition that must never pass silently.
    RECONCILE_MISMATCH = "reconcile_mismatch"
    MISSING_STOP = "missing_stop"
    UNHANDLED_ERROR = "unhandled_error"
    HALT = "halt"


class AlertManager:
    """Rate-limited fan-out to console, log file, email and webhook."""

    #: A repeated halt is worth hearing more than once an hour.
    CRITICAL_RATE_DIVISOR = 3

    def __init__(
        self,
        settings: Optional[dict[str, Any]] = None,
        rate_limit_minutes: int = 15,
        sink: Optional[Callable[[AlertLevel, str, str], None]] = None,
        clock: Callable[[], float] = time.monotonic,
        trading_logger=None,
    ) -> None:
        self.settings = dict(settings or {})
        self.rate_limit_minutes = rate_limit_minutes
        self.trading_logger = trading_logger
        self._last_sent: dict[str, float] = {}
        self._lock = threading.Lock()
        self._clock = clock
        self._sink = sink
        self.sent: list[dict[str, Any]] = []
        self.suppressed = 0

        threshold = str(self.settings.get("alert_min_level", "warning")).lower()
        self.min_delivery_level = AlertLevel(threshold) if threshold in (
            "info", "warning", "critical") else AlertLevel.WARNING

        self.large_pnl_pct = float(self.settings.get("large_pnl_pct", 0.02))

    # -- configuration ------------------------------------------------------

    @property
    def email_to(self) -> Optional[str]:
        return self.settings.get("alert_email") or os.getenv("ALERT_EMAIL")

    @property
    def webhook_url(self) -> Optional[str]:
        return self.settings.get("alert_webhook") or os.getenv("ALERT_WEBHOOK_URL")

    # -- rate limiting ------------------------------------------------------

    def _window_seconds(self, level: AlertLevel) -> float:
        minutes = self.rate_limit_minutes
        if level is AlertLevel.CRITICAL:
            minutes = minutes / self.CRITICAL_RATE_DIVISOR
        return minutes * 60.0

    def is_rate_limited(self, dedupe_key: str, level: AlertLevel = AlertLevel.WARNING) -> bool:
        with self._lock:
            last = self._last_sent.get(dedupe_key)
        if last is None:
            return False
        return (self._clock() - last) < self._window_seconds(level)

    # -- sending ------------------------------------------------------------

    def send(
        self,
        level: AlertLevel,
        subject: str,
        body: str,
        dedupe_key: Optional[str] = None,
        trigger: Optional[AlertTrigger] = None,
    ) -> bool:
        """Send an alert. Returns False if suppressed by the rate limit.

        `dedupe_key` defaults to the trigger, which is what makes the limit "one
        per event type per 15 minutes" rather than one per distinct wording.
        """
        key = dedupe_key or (trigger.value if trigger else subject)
        if self.is_rate_limited(key, level):
            self.suppressed += 1
            logger.debug("alert suppressed by rate limit: %s", key)
            return False

        with self._lock:
            self._last_sent[key] = self._clock()

        record = {
            "level": level.value, "subject": subject, "body": body,
            "key": key, "trigger": trigger.value if trigger else None,
        }
        self.sent.append(record)

        # 1. console
        if self._sink is not None:
            try:
                self._sink(level, subject, body)
            except Exception as exc:      # pragma: no cover - custom sink failure
                logger.warning("alert sink failed: %s", exc)
        else:
            logger.log(
                {AlertLevel.INFO: logging.INFO, AlertLevel.WARNING: logging.WARNING,
                 AlertLevel.CRITICAL: logging.CRITICAL}[level],
                "ALERT [%s] %s\n%s", level.value.upper(), subject, body,
            )

        # 2. log file, via the structured logger's alerts stream
        if self.trading_logger is not None:
            try:
                from monitoring.logger import EventType

                self.trading_logger.log_event(
                    EventType.ALERT_SENT, "", alert_level=level.value,
                    alert_subject=subject, alert_body=body,
                    alert_trigger=record["trigger"],
                )
            except Exception as exc:      # pragma: no cover
                logger.warning("alert log write failed: %s", exc)

        # 3 and 4. email and webhook, above the delivery threshold only.
        if _LEVEL_ORDER[level] >= _LEVEL_ORDER[self.min_delivery_level]:
            if self.email_to:
                self.send_email(subject, body)
            if self.webhook_url:
                self.send_webhook(
                    {"level": level.value, "subject": subject, "body": body,
                     "trigger": record["trigger"]}
                )
        return True

    def send_email(self, subject: str, body: str) -> None:
        """SMTP settings come from the environment. Never raises."""
        host = self.settings.get("smtp_host") or os.getenv("SMTP_HOST")
        if not host or not self.email_to:
            return
        try:
            message = EmailMessage()
            message["Subject"] = f"[regime-trader] {subject}"
            message["From"] = os.getenv("SMTP_FROM", self.email_to)
            message["To"] = self.email_to
            message.set_content(body)

            port = int(self.settings.get("smtp_port") or os.getenv("SMTP_PORT", 587))
            user, password = os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD")

            with smtplib.SMTP(host, port, timeout=10) as smtp:
                smtp.starttls()
                if user and password:
                    smtp.login(user, password)
                smtp.send_message(message)
        except Exception as exc:
            logger.warning("alert email failed (%s). Console alert still delivered.", exc)

    def send_webhook(self, payload: dict) -> None:
        """POST to the configured webhook. Never raises.

        urllib rather than requests, so alerting has no dependency that could be
        missing on the machine where it matters.
        """
        url = self.webhook_url
        if not url:
            return
        try:
            import json
            import urllib.request

            request = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=10).close()
        except Exception as exc:
            logger.warning("alert webhook failed (%s). Console alert still delivered.", exc)

    # -- the seven triggers -------------------------------------------------

    def alert_regime_change(self, previous: str, current: str, confidence: float,
                            confirmed: bool = True) -> bool:
        """INFO. Routine: this is the system working, not failing."""
        return self.send(
            AlertLevel.INFO,
            f"Regime: {previous} -> {current}",
            f"Confidence {confidence:.0%}, "
            f"{'confirmed' if confirmed else 'awaiting confirmation'}.",
            trigger=AlertTrigger.REGIME_CHANGE,
        )

    def alert_breaker_triggered(self, breaker: str, drawdown: float, equity: float) -> bool:
        """CRITICAL. The halt breaker needs manual intervention to clear, so
        nobody finding out is the failure mode this prevents."""
        return self.send(
            AlertLevel.CRITICAL,
            f"Circuit breaker: {breaker}",
            f"Drawdown {drawdown:.2%}, equity ${equity:,.2f}.\n"
            f"If this wrote trading_halted.lock, the system will not trade again "
            f"until you delete that file by hand.",
            dedupe_key=f"{AlertTrigger.CIRCUIT_BREAKER.value}:{breaker}",
            trigger=AlertTrigger.CIRCUIT_BREAKER,
        )

    def alert_large_pnl(self, pnl: float, pnl_pct: float, equity: float) -> bool:
        """WARNING. Fires on a move in either direction.

        A large gain is as much a reason to look as a large loss: unless the
        system was meant to make 4% today, something is sized wrong.
        """
        direction = "gain" if pnl >= 0 else "loss"
        return self.send(
            AlertLevel.WARNING,
            f"Large daily {direction}: {pnl_pct:+.2%}",
            f"{pnl:+,.2f} on equity ${equity:,.2f}, past the "
            f"{self.large_pnl_pct:.1%} notification threshold.",
            trigger=AlertTrigger.LARGE_PNL,
        )

    def alert_data_feed_down(self, detail: str = "") -> bool:
        return self.send(
            AlertLevel.CRITICAL, "Data feed down",
            f"{detail}\nSignals are paused. Stops already resting at the broker "
            f"are unaffected.",
            trigger=AlertTrigger.DATA_FEED_DOWN,
        )

    def alert_broker_down(self, error: str) -> bool:
        """API lost. Distinct from the data feed: this one means orders and stop
        modifications may not be reaching the broker at all."""
        return self.send(
            AlertLevel.CRITICAL, "Broker API unreachable",
            f"Repeated failures talking to Alpaca: {error}\n"
            f"No new orders will be placed. Stops already resting at the broker "
            f"remain active.",
            trigger=AlertTrigger.API_LOST,
        )

    def alert_hmm_retrained(self, n_states: int, reason: str = "",
                            previous_states: Optional[int] = None) -> bool:
        """INFO, unless the state count changed.

        A refit that lands on a different number of regimes has redrawn the map
        the allocator reads, and that is worth more than a routine note.
        """
        changed = previous_states is not None and previous_states != n_states
        return self.send(
            AlertLevel.WARNING if changed else AlertLevel.INFO,
            f"HMM retrained: {n_states} states",
            f"{reason}\n" + (
                f"State count changed from {previous_states} to {n_states}. "
                f"Every regime's strategy mapping has been rebuilt."
                if changed else "State count unchanged."
            ),
            trigger=AlertTrigger.HMM_RETRAINED,
        )

    def alert_flicker_exceeded(self, flicker_rate: int, threshold: int,
                               window: int = 20) -> bool:
        """WARNING. The model is unsure; position size is already halved."""
        return self.send(
            AlertLevel.WARNING, f"Regime flickering: {flicker_rate}/{window}",
            f"{flicker_rate} raw regime changes in the last {window} bars, past the "
            f"threshold of {threshold}. The model cannot decide what regime this is, "
            f"so position sizes are halved until it settles.",
            trigger=AlertTrigger.FLICKER_EXCEEDED,
        )

    # -- beyond the spec, but each must never pass silently -----------------

    def alert_reconcile_mismatch(self, discrepancies: list[str]) -> bool:
        return self.send(
            AlertLevel.CRITICAL, "Position reconciliation mismatch",
            "The broker and the tracker disagree about what is held:\n  "
            + "\n  ".join(discrepancies),
            trigger=AlertTrigger.RECONCILE_MISMATCH,
        )

    def alert_missing_stop(self, symbols: list[str]) -> bool:
        """An open position with no stop is the one thing this system promises
        cannot happen. If it happens, it is worth an alert every time."""
        return self.send(
            AlertLevel.CRITICAL, "Open position with no stop",
            "These positions have no protective stop:\n  " + "\n  ".join(symbols)
            + "\nPlace one manually or close the position.",
            dedupe_key=f"{AlertTrigger.MISSING_STOP.value}:" + ",".join(sorted(symbols)),
            trigger=AlertTrigger.MISSING_STOP,
        )

    def alert_unhandled_error(self, where: str, error: str, traceback_text: str = "") -> bool:
        return self.send(
            AlertLevel.CRITICAL, f"Unhandled error in {where}",
            f"{error}\n\n{traceback_text}",
            dedupe_key=f"{AlertTrigger.UNHANDLED_ERROR.value}:{where}",
            trigger=AlertTrigger.UNHANDLED_ERROR,
        )

    def alert_halt(self, reason: str, equity: float) -> bool:
        return self.send(
            AlertLevel.CRITICAL, "System halted",
            f"{reason}\nEquity at halt: ${equity:,.2f}",
            trigger=AlertTrigger.HALT,
        )
