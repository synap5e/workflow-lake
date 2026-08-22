"""Politeness layer shared by every probe.

Per-host token bucket, a self-identifying User-Agent, exponential backoff on
429/5xx, and a
hard stop when a host answers 403 repeatedly (we stop probing rather than try to
look like a browser). Also counts bytes so bandwidth claims in FINDINGS.md are
measured, not guessed.
"""

from __future__ import annotations

import os
import random
import socket
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx

UA = os.environ.get(
    "LAKE_UA",
    "comfy-workflow-lake/0.1",
)


_ipv6_state: bool | None = None
_ipv6_lock = threading.Lock()


def ipv6_usable(probe: str = "2606:4700:4700::1111", timeout: float = 2.0) -> bool:
    """Is IPv6 egress actually working here?

    httpx does not do Happy Eyeballs: on a dual-stack host whose IPv6 egress is
    black-holed it connects to the AAAA address and sits there until the timeout,
    turning a 50 ms fetch into a 40 second one. curl hides the same fault because
    it races the two families, which makes this astonishingly easy to misdiagnose
    as "the source is throttling us".

    So probe once per process and force IPv4 when v6 is dead. Override with
    LAKE_IP_FAMILY=4|6|auto.
    """
    global _ipv6_state
    forced = os.environ.get("LAKE_IP_FAMILY", "auto").lower()
    if forced in ("4", "ipv4"):
        return False
    if forced in ("6", "ipv6"):
        return True
    with _ipv6_lock:
        if _ipv6_state is None:
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            try:
                sock.connect((probe, 443))
                _ipv6_state = True
            except OSError:
                _ipv6_state = False
            finally:
                sock.close()
        return _ipv6_state


class HostBlocked(RuntimeError):
    """Raised when a host has told us off enough times to stop."""


@runtime_checkable
class LatchSink(Protocol):
    """Somewhere refusals persist across pod restarts.

    In-process counting is enough for a probe run by hand; under a CronJob the
    counter dies with the pod, so a source that starts refusing us gets hit again
    at the next tick forever. `pipeline` supplies a Postgres-backed
    implementation; tests supply a fake; a bare probe supplies None.
    """

    def refused(self, host: str, status: int) -> bool:
        """Record a refusal. Return True if the host is now latched."""
        ...

    def succeeded(self, host: str) -> None: ...


@dataclass
class Stats:
    requests: int = 0
    bytes_down: int = 0
    status: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    retries: int = 0


class PoliteClient:
    def __init__(
        self,
        *,
        rps: float = 1.5,
        timeout: float = 45.0,
        max_retries: int = 3,
        forbidden_budget: int = 8,
        user_agent: str | None = None,
        latch: LatchSink | None = None,
    ) -> None:
        self._min_gap = 1.0 / rps
        self._next_ok: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
        self._forbidden: dict[str, int] = defaultdict(int)
        self._forbidden_budget = forbidden_budget
        # Where refusals go to outlive this process. None = in-process only,
        # which is right for a one-off probe and wrong for a CronJob.
        self._latch = latch
        self._max_retries = max_retries
        self.stats = Stats()
        # local_address="0.0.0.0" pins the socket family to IPv4.
        transport = httpx.HTTPTransport(
            retries=1, local_address=None if ipv6_usable() else "0.0.0.0"
        )
        self.client = httpx.Client(
            headers={"User-Agent": user_agent or UA, "Accept-Encoding": "identity"},
            follow_redirects=True,
            # Separate connect timeout so a dead route fails fast rather than
            # burning the whole request budget on a TCP handshake.
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
            http2=False,
        )

    def _gate(self, host: str) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                due = self._next_ok[host]
                if now >= due:
                    self._next_ok[host] = now + self._min_gap
                    return
                wait = due - now
            time.sleep(wait)

    def get(self, url: str, **kw) -> httpx.Response:
        host = urlsplit(url).netloc
        if self._forbidden[host] >= self._forbidden_budget:
            raise HostBlocked(f"{host} returned 403/401 {self._forbidden[host]}x; stopping")
        delay = 2.0
        last: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            self._gate(host)
            try:
                resp = self.client.get(url, **kw)
            except httpx.HTTPError as exc:
                if attempt == self._max_retries:
                    raise
                self.stats.retries += 1
                time.sleep(delay + random.random())
                delay *= 2
                _ = exc
                continue
            self.stats.requests += 1
            self.stats.status[resp.status_code] += 1
            self.stats.bytes_down += len(resp.content)
            last = resp
            if resp.status_code in (401, 403, 429):
                self._forbidden[host] += 1
                if self._latch is not None and self._latch.refused(host, resp.status_code):
                    raise HostBlocked(
                        f"{host} latched after repeated HTTP {resp.status_code}; "
                        f"clear it with `lake unlatch {host}` once you know why"
                    )
                if resp.status_code in (401, 403):
                    return resp
            if self._latch is not None and resp.status_code < 400:
                self._latch.succeeded(host)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self._max_retries:
                    return resp
                retry_after = resp.headers.get("retry-after")
                sleep_for = float(retry_after) if (retry_after or "").isdigit() else delay
                self.stats.retries += 1
                time.sleep(sleep_for + random.random())
                delay *= 2
                continue
            return resp
        assert last is not None
        return last

    def get_range(self, url: str, start: int, end_inclusive: int, **kw) -> httpx.Response:
        headers = dict(kw.pop("headers", {}))
        headers["Range"] = f"bytes={start}-{end_inclusive}"
        return self.get(url, headers=headers, **kw)

    def get_tail(self, url: str, length: int, **kw) -> httpx.Response:
        headers = dict(kw.pop("headers", {}))
        headers["Range"] = f"bytes=-{length}"
        return self.get(url, headers=headers, **kw)

    def close(self) -> None:
        self.client.close()


def total_size_from(resp: httpx.Response) -> int | None:
    cr = resp.headers.get("content-range")
    if cr and "/" in cr:
        tail = cr.rsplit("/", 1)[1]
        if tail.isdigit():
            return int(tail)
    cl = resp.headers.get("content-length")
    if resp.status_code == 200 and cl and cl.isdigit():
        return int(cl)
    return None
