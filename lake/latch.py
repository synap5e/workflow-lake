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


class PostgresLatch:
    """Refusals that outlive the pod."""

    def __init__(self, frontier: Frontier, threshold: int = DEFAULT_THRESHOLD) -> None:
        self.frontier = frontier
        self.threshold = threshold

    def refused(self, host: str, status: int) -> bool:
        return self.frontier.record_refusal(host, status, self.threshold)

    def succeeded(self, host: str) -> None:
        self.frontier.record_success(host)


class Latched(RuntimeError):
    """A job refused to start because a host it needs is latched."""


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
