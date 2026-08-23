"""The Postgres-backed latch sink, and the gate every job runs first.

Split from `lake/db.py` so `lake/polite.py` depends on a small protocol rather
than on the frontier, which keeps the probes in `probes/` runnable with no
database.
"""

from __future__ import annotations

from .db import Frontier

# Consecutive 401/403/429 from one host before we stop. Three rather than one:
# a single 403 is often a CDN blip or one bad object, and latching on it would
# make the crawler stop for reasons a human then has to dismiss. Three in a row
# with no success between them is the source talking.
DEFAULT_THRESHOLD = 3

# Consecutive transient 5xx before backing a host off, and the first delay.
# Three for the same reason as above; 15 minutes because that is one discover
# tick, so the first backoff simply skips a turn.
TRANSIENT_THRESHOLD = 3
TRANSIENT_BASE_SECONDS = 900


class PostgresLatch:
    """Refusals that outlive the pod."""

    def __init__(self, frontier: Frontier, threshold: int = DEFAULT_THRESHOLD) -> None:
        self.frontier = frontier
        self.threshold = threshold

    def refused(self, host: str, status: int) -> bool:
        return self.frontier.record_refusal(host, status, self.threshold)

    def succeeded(self, host: str) -> None:
        self.frontier.record_success(host)
        # A success ends a transient outage as well as a refusal streak. This is
        # what makes the backoff self-clearing rather than something a human has
        # to notice and undo.
        self.frontier.clear_transient(host)

    def transient(self, host: str, status: int) -> int:
        """Count a transient 5xx. Returns seconds backed off, 0 if not yet."""
        return self.frontier.record_transient(
            host, status, TRANSIENT_THRESHOLD, TRANSIENT_BASE_SECONDS
        )


class Latched(RuntimeError):
    """A job refused to start because a host it needs is latched."""


class BackingOff(RuntimeError):
    """A job declined to start because a host is in transient backoff.

    Deliberately not a subclass of `Latched`: a latch is an incident that exits
    2 and waits for a human, while this is the crawler behaving correctly and
    exits 0. Marking the CronJob Failed for an expected quiet period would be
    the same category error as alerting on a healthy sawtooth.
    """


def gate(frontier: Frontier, hosts: list[str]) -> None:
    """Refuse to start while any required host is latched.

    Called at the top of every job that makes outbound requests. This is the
    half that matters: recording a latch is useless if the next tick ignores it.
    """
    for host in hosts:
        row = frontier.latched(host)
        if row is not None:
            raise Latched(
                f"{host} latched at {row['latched_at']:%Y-%m-%d %H:%M} — {row['reason']}. "
                f"Investigate, then `lake unlatch {host}`."
            )
    for host in hosts:
        row = frontier.backing_off(host)
        if row is not None:
            raise BackingOff(
                f"{host} returned {row['consecutive_transient']} consecutive "
                f"HTTP {row['last_status']}; backing off {row['seconds_left']}s more. "
                f"Clears itself on the first success — no action needed."
            )
