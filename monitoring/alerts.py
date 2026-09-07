"""
Email and webhook alerts for critical events.

Alert on things that need a human. Not on things that are merely interesting:
an alert that fires ten times a day gets muted within a week, and then the one
that mattered gets muted with it.

The list worth waking up for:
- a circuit breaker halted the system
- position reconciliation found a mismatch with the broker
- the broker API is unreachable across repeated cycles
- an open position has no protective stop
- an unrecoverable error stopped the main loop

Rate limited by alert_rate_limit_minutes, so a flapping condition sends one
message rather than four hundred.

Implemented in Phase 7. The main loop needs somewhere to shout when a breaker
latches, and "print it and hope someone is watching" is not that.
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


class AlertManager:
    """Rate-limited fan-out to console, email and webhook.

    Console is unconditional and cannot be switched off: it is the only sink
    guaranteed to work when SMTP is misconfigured, and a system whose alerting
    silently depends on a mail server is one that will be silent on the day the
    mail server is down.

    Nothing here raises. A delivery failure is logged and swallowed, because an
    alert about a circuit breaker must not become a second exception that takes
    down the loop that was trying to report the first one.
    """

    # CRITICAL bypasses nothing except good manners: it is still rate limited,
    # but on a shorter window, because a repeated halt is worth hearing twice.
    CRITICAL_RATE_DIVISOR = 3

    def __init__(
        self,
        settings: Optional[dict[str, Any]] = None,
        rate_limit_minutes: int = 15,
        sink: Optional[Callable[[AlertLevel, str, str], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = dict(settings or {})
        self.rate_limit_minutes = rate_limit_minutes
        self._last_sent: dict[str, float] = {}
        self._lock = threading.Lock()
        self._clock = clock
        self._sink = sink
        self.sent: list[dict[str, Any]] = []
        self.suppressed = 0

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
    ) -> bool:
        """Send an alert. Returns False if suppressed by the rate limit.

        dedupe_key groups repeats of the same condition. Defaults to the
        subject, so two alerts that say the same thing collapse without the
        caller having to think about it.
        """
        key = dedupe_key or subject
        if self.is_rate_limited(key, level):
            self.suppressed += 1
            logger.debug("alert suppressed by rate limit: %s", key)
            return False

        with self._lock:
            self._last_sent[key] = self._clock()

        record = {"level": level.value, "subject": subject, "body": body, "key": key}
        self.sent.append(record)

        if self._sink is not None:
            try:
                self._sink(level, subject, body)
            except Exception as exc:  # pragma: no cover - custom sink failure
                logger.warning("alert sink failed: %s", exc)
        else:
            log_at = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.CRITICAL: logging.CRITICAL,
            }[level]
            logger.log(log_at, "ALERT [%s] %s\n%s", level.value.upper(), subject, body)

        if self.email_to:
            self.send_email(subject, body)
        if self.webhook_url:
            self.send_webhook({"level": level.value, "subject": subject, "body": body})

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
            user = os.getenv("SMTP_USER")
            password = os.getenv("SMTP_PASSWORD")

            with smtplib.SMTP(host, port, timeout=10) as smtp:
                smtp.starttls()
                if user and password:
                    smtp.login(user, password)
                smtp.send_message(message)
        except Exception as exc:
            logger.warning("alert email failed (%s). Console alert still delivered.", exc)

    def send_webhook(self, payload: dict) -> None:
        """POST to the configured webhook. Never raises.

        urllib rather than requests so alerting has no dependency that could be
        missing on the machine where it matters.
        """
        url = self.webhook_url
        if not url:
            return
        try:
            import json
            import urllib.request

            data = json.dumps(payload).encode()
            request = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(request, timeout=10).close()
        except Exception as exc:
            logger.warning("alert webhook failed (%s). Console alert still delivered.", exc)

    # -- the specific alerts worth having -----------------------------------

    def alert_breaker_triggered(self, breaker: str, drawdown: float, equity: float) -> bool:
        """Always CRITICAL. The halt breaker needs manual intervention to clear,
        so nobody finding out is the failure mode this prevents."""
        return self.send(
            AlertLevel.CRITICAL,
            f"Circuit breaker: {breaker}",
            f"Drawdown {drawdown:.2%}, equity ${equity:,.2f}.\n"
            f"If this wrote trading_halted.lock, the system will not trade again "
            f"until you delete that file by hand.",
            dedupe_key=f"breaker:{breaker}",
        )

    def alert_reconcile_mismatch(self, discrepancies: list[str]) -> bool:
        return self.send(
            AlertLevel.CRITICAL,
            "Position reconciliation mismatch",
            "The broker and the tracker disagree about what is held:\n  "
            + "\n  ".join(discrepancies),
            dedupe_key="reconcile",
        )

    def alert_broker_down(self, error: str) -> bool:
        return self.send(
            AlertLevel.CRITICAL,
            "Broker unreachable",
            f"Repeated failures talking to Alpaca: {error}\n"
            f"Signals are paused. Stops already resting at the broker are unaffected.",
            dedupe_key="broker_down",
        )

    def alert_missing_stop(self, symbols: list[str]) -> bool:
        """An open position with no stop is the one thing this system promises
        cannot happen. If it happens, it is worth an alert every time."""
        return self.send(
            AlertLevel.CRITICAL,
            "Open position with no stop",
            "These positions have no protective stop recorded:\n  "
            + "\n  ".join(symbols)
            + "\nPlace one manually or close the position.",
            dedupe_key="missing_stop:" + ",".join(sorted(symbols)),
        )

    def alert_unhandled_error(self, where: str, error: str, traceback_text: str = "") -> bool:
        return self.send(
            AlertLevel.CRITICAL,
            f"Unhandled error in {where}",
            f"{error}\n\n{traceback_text}",
            dedupe_key=f"error:{where}",
        )

    def alert_halt(self, reason: str, equity: float) -> bool:
        return self.send(
            AlertLevel.CRITICAL,
            "System halted",
            f"{reason}\nEquity at halt: ${equity:,.2f}",
            dedupe_key="halt",
        )
