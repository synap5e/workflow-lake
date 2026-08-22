"""Latch tests.

The latch exists because the in-process stop-budget dies with a CronJob pod, so
what matters is that a refusal survives the process and that the next run
actually refuses to start. Both halves are tested; either alone is useless.
"""

from __future__ import annotations

import httpx
import pytest

from lake.latch import Latched, gate
from lake.polite import HostBlocked, PoliteClient


class FakeFrontier:
    """The persistence the latch needs, without a database."""

    def __init__(self, threshold: int = 3) -> None:
        self.consecutive: dict[str, int] = {}
        self.latches: dict[str, dict] = {}
        self.threshold = threshold

    def record_refusal(self, host, status, threshold, detail="") -> bool:
        self.consecutive[host] = self.consecutive.get(host, 0) + 1
        if self.consecutive[host] < threshold:
            return False
        self.latches.setdefault(
            host,
            {
                "latched_at": __import__("datetime").datetime(2026, 8, 22),
                "reason": f"{self.consecutive[host]} consecutive HTTP {status}",
            },
        )
        return True

    def record_success(self, host) -> None:
        self.consecutive[host] = 0

    def latched(self, host):
        return self.latches.get(host)


class Sink:
    def __init__(self, frontier: FakeFrontier) -> None:
        self.frontier = frontier

    def refused(self, host, status) -> bool:
        return self.frontier.record_refusal(host, status, self.frontier.threshold)

    def succeeded(self, host) -> None:
        self.frontier.record_success(host)


def client_returning(status: int, frontier: FakeFrontier) -> PoliteClient:
    c = PoliteClient(rps=1000, max_retries=0, latch=Sink(frontier))
    c.client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(status, content=b""))
    )
    return c


@pytest.mark.parametrize("status", [401, 403, 429])
def test_repeated_refusal_latches_and_stops_the_run(status: int) -> None:
    f = FakeFrontier(threshold=3)
    c = client_returning(status, f)
    c.get("https://example.test/a")
    c.get("https://example.test/b")
    with pytest.raises(HostBlocked, match="latched"):
        c.get("https://example.test/c")
    assert "example.test" in f.latches


def test_a_single_refusal_does_not_latch() -> None:
    """One 403 is often a blip or one bad object; latching on it would make a
    human dismiss the alarm, which is how alarms stop being read."""
    f = FakeFrontier(threshold=3)
    client_returning(403, f).get("https://example.test/a")
    assert f.latches == {}


def test_success_resets_the_tally() -> None:
    """Refusals must be *consecutive*, or a slow trickle latches after days."""
    f = FakeFrontier(threshold=3)
    bad, good = client_returning(403, f), client_returning(200, f)
    bad.get("https://example.test/a")
    bad.get("https://example.test/b")
    good.get("https://example.test/ok")
    assert f.consecutive["example.test"] == 0
    bad.get("https://example.test/c")
    assert f.latches == {}


def test_gate_blocks_the_next_run() -> None:
    """The half that matters: recording a latch is useless if the next tick
    ignores it."""
    f = FakeFrontier()
    f.latches["civitai.com"] = {
        "latched_at": __import__("datetime").datetime(2026, 8, 22),
        "reason": "3 consecutive HTTP 403",
    }
    with pytest.raises(Latched, match="lake unlatch civitai.com"):
        gate(f, ["civitai.com", "image.civitai.com"])


def test_gate_passes_when_clear() -> None:
    gate(FakeFrontier(), ["civitai.com", "image.civitai.com"])


def test_latch_is_per_host() -> None:
    """One rude source must not stop us crawling every other source."""
    f = FakeFrontier(threshold=1)
    with pytest.raises(HostBlocked):  # latches on the first refusal at threshold 1
        client_returning(403, f).get("https://blocked.test/a")
    gate(f, ["other.test"])  # unaffected
    with pytest.raises(Latched):
        gate(f, ["blocked.test"])
