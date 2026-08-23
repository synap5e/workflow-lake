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

    # gate() checks transient backoff as well as the latch.
    def backing_off(self, host):
        return None

    def clear_transient(self, host):
        pass


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


class BackoffFrontier:
    """Frontier surface for transient backoff, with the real self-clearing rule."""

    def __init__(self) -> None:
        self.consecutive = 0
        self.until_seconds = 0
        self.cleared = 0

    def record_transient(self, host, status, threshold, base_seconds):
        self.consecutive += 1
        if self.consecutive < threshold:
            return 0
        self.until_seconds = min(base_seconds * (2 ** (self.consecutive - threshold)), 86400)
        return self.until_seconds

    def record_success(self, host):
        pass

    def clear_transient(self, host):
        self.consecutive = 0
        self.until_seconds = 0
        self.cleared += 1

    def latched(self, host):
        return None

    def backing_off(self, host):
        if not self.until_seconds:
            return None
        return {
            "consecutive_transient": self.consecutive,
            "last_status": 503,
            "seconds_left": self.until_seconds,
        }


def test_transient_5xx_backs_off_only_after_a_streak_and_then_doubles() -> None:
    """Civitai returned 503 for four hours and we knocked every 15 minutes.

    One 503 must not back anything off — sources blip. A streak must, and the
    delay has to grow, or a long outage is just a slower version of knocking.
    """
    from lake.latch import TRANSIENT_BASE_SECONDS, PostgresLatch

    f = BackoffFrontier()
    latch = PostgresLatch(f)

    assert latch.transient("civitai.com", 503) == 0, "one blip is not an outage"
    assert latch.transient("civitai.com", 503) == 0
    assert latch.transient("civitai.com", 503) == TRANSIENT_BASE_SECONDS
    assert latch.transient("civitai.com", 503) == TRANSIENT_BASE_SECONDS * 2
    assert latch.transient("civitai.com", 503) == TRANSIENT_BASE_SECONDS * 4


def test_backoff_clears_itself_on_the_first_success() -> None:
    """The property that makes this NOT the latch.

    A latch is an incident a human clears. Transient overload resolves on its
    own — as this one did, after four hours — so requiring a human would have
    stopped the crawler until someone noticed a problem that had already fixed
    itself.
    """
    from lake.latch import PostgresLatch

    f = BackoffFrontier()
    latch = PostgresLatch(f)
    for _ in range(4):
        latch.transient("civitai.com", 503)
    assert f.backing_off("civitai.com") is not None

    latch.succeeded("civitai.com")

    assert f.backing_off("civitai.com") is None
    assert f.cleared == 1


def test_backing_off_exits_zero_while_latched_exits_two() -> None:
    """Different operational states must not look the same to the kubelet.

    Exit 2 means a human is needed. A backoff tick is the crawler working, so
    it exits 0 — a Failed Job for an expected quiet period is noise someone has
    to triage, which is how the real 503 outage stayed invisible in a wall of
    red for four hours.
    """
    from lake.latch import BackingOff, Latched, gate

    f = BackoffFrontier()
    for _ in range(4):
        f.record_transient("civitai.com", 503, 3, 900)

    try:
        gate(f, ["civitai.com"])
    except BackingOff as exc:
        assert "clears itself" in str(exc).lower()
    else:
        raise AssertionError("gate must refuse to start while backing off")

    assert not issubclass(BackingOff, Latched), "the exit codes depend on these being distinct"
