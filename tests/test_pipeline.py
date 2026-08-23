"""Pipeline tests that need no network and no database.

The fetch policies are the expensive things to get wrong (they decide bandwidth
and coverage), so they are tested against a stub transport rather than left to be
discovered in production.
"""

from __future__ import annotations

import gzip
import json

import httpx
import pytest
from fixtures import SAVE_GRAPH, mp4, png

from lake.capture import fetch_bytes, should_try_api
from lake.polite import PoliteClient
from lake.storage import BlobWriter, LocalStore, ManifestWriter, blob_key, read_manifest


class Served:
    """Serves one blob over ranged GETs and counts what was actually requested."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.ranges: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("Range", "")
        self.ranges.append(rng)
        total = len(self.body)
        if rng.startswith("bytes=-"):
            start, end = max(0, total - int(rng[7:])), total - 1
        elif rng.startswith("bytes="):
            first, _, last = rng[6:].partition("-")
            start = int(first)
            end = min(int(last), total - 1) if last else total - 1
        else:
            return httpx.Response(200, content=self.body)
        if start >= total:
            return httpx.Response(416)
        chunk = self.body[start : end + 1]
        return httpx.Response(
            206,
            content=chunk,
            headers={"Content-Range": f"bytes {start}-{start + len(chunk) - 1}/{total}"},
        )


def client_for(served: Served) -> PoliteClient:
    client = PoliteClient(rps=1000)
    client.client = httpx.Client(transport=httpx.MockTransport(served.handler))
    return client


def test_png_is_answered_from_one_head_window() -> None:
    served = Served(png() + b"\x00" * 5_000_000)
    client = client_for(served)
    result = fetch_bytes(client, "https://example.test/a.png")
    assert json.loads(result.workflow) == SAVE_GRAPH
    assert len(served.ranges) == 1, "a PNG with metadata up front needs exactly one range"
    assert result.bytes_fetched <= 128 * 1024


def test_negative_png_costs_one_window_and_no_tail() -> None:
    served = Served(png(workflow=None, prompt=None) + b"\x00" * 2_000_000)
    client = client_for(served)
    result = fetch_bytes(client, "https://example.test/a.png")
    assert not result.has_payload
    assert served.ranges == ["bytes=0-131071"], "must not chase a tail on a plain PNG"


def test_video_falls_back_to_the_tail() -> None:
    served = Served(mp4(faststart=False))
    client = client_for(served)
    result = fetch_bytes(client, "https://example.test/a.mp4")
    assert json.loads(result.workflow) == SAVE_GRAPH
    assert result.from_tail
    assert result.channel == "bytes_tail"
    assert served.ranges[-1].startswith("bytes=-"), "the last request must be a tail range"


def test_faststart_video_needs_no_tail() -> None:
    served = Served(mp4(faststart=True))
    client = client_for(served)
    result = fetch_bytes(client, "https://example.test/a.mp4")
    assert result.workflow is not None
    assert not result.from_tail
    assert not any(r.startswith("bytes=-") for r in served.ranges)


def test_escalation_continues_rather_than_refetching() -> None:
    """Re-fetching from byte 0 on escalation can cost more than the whole file;
    the second window must start where the first stopped."""
    big = json.dumps({**SAVE_GRAPH, "notes": "x" * 300_000})
    served = Served(png(workflow=big))
    client = client_for(served)
    result = fetch_bytes(client, "https://example.test/big.png")
    assert result.workflow is not None
    assert len(served.ranges) == 2
    assert served.ranges[0] == "bytes=0-131071"
    assert served.ranges[1].startswith("bytes=131072-")


@pytest.mark.parametrize(
    "has_workflow,has_meta,expected",
    [(True, True, False), (True, False, False), (False, True, True), (False, False, False)],
)
def test_api_channel_is_gated_on_hasmeta(has_workflow, has_meta, expected) -> None:
    """Measured: gating the API call on `hasMeta` cut calls by up to 81% and lost
    no workflows, because Civitai sets the flag exactly when it has the data."""
    from lake.capture import ChannelResult

    result = ChannelResult(channel="bytes_head", workflow="{}" if has_workflow else None)
    assert should_try_api(result, {"has_meta": has_meta}) is expected


def test_blobs_are_content_addressed_and_deduped(tmp_path) -> None:
    class FakeFrontier:
        def __init__(self) -> None:
            self.seen: set[bytes] = set()

        def blob_exists(self, sha: bytes) -> bool:
            return sha in self.seen

        def record_blob(self, sha, **kw) -> None:
            self.seen.add(sha)

    store = LocalStore(tmp_path)
    writer = BlobWriter(store, FakeFrontier(), keep_prefix_days=90)
    sha_a, key_a, _ = writer.put(b'{"a":1}', "workflow")
    sha_b, key_b, _ = writer.put(b'{"a":1}', "workflow")
    assert (sha_a, key_a) == (sha_b, key_b)
    assert writer.written == 1 and writer.deduped == 1
    assert gzip.decompress(store.get(key_a)) == b'{"a":1}'
    assert key_a == blob_key(bytes.fromhex(sha_a), "workflow")


def test_manifest_round_trips(tmp_path) -> None:
    store = LocalStore(tmp_path)
    writer = ManifestWriter(store, "civitai", "run-1")
    writer.write({"capture_id": "a", "source_meta": {"nested": [1, 2]}})
    key = writer.commit()
    assert read_manifest(store, key)[0]["source_meta"] == {"nested": [1, 2]}


def test_empty_manifest_writes_nothing(tmp_path) -> None:
    store = LocalStore(tmp_path)
    assert ManifestWriter(store, "civitai", "run-2").commit() is None


def test_auth_failure_is_not_mistaken_for_a_missing_artifact() -> None:
    """A 401 marked as `skipped: gone` would march the entire queue to skipped
    and report a clean run — the same exit-0-while-doing-nothing shape that has
    already bitten this project twice."""
    from lake import civitai

    served = Served(b"")
    client = client_for(served)
    client.client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": "no"}))
    )
    with pytest.raises(civitai.AuthRequired):
        civitai.image_get(client, 123)


def test_a_genuine_404_still_returns_none() -> None:
    from lake import civitai

    client = PoliteClient(rps=1000, max_retries=0)
    client.client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(404, json={"error": "gone"}))
    )
    assert civitai.image_get(client, 123) is None


class LeaseFrontier:
    """Enough Frontier to exercise the lease lifecycle."""

    def __init__(self, n: int) -> None:
        self.leased = [{"artifact_id": str(i), "prefilter": {}} for i in range(n)]
        self.finished: dict[str, str] = {}
        self.released: list[str] = []
        self.runs: list[str] = []

    def start_run(self, job, source):
        self.runs.append(job)
        return "run-1"

    def finish_run(self, run_id, **kw):
        self.last_run = kw

    def claim(self, source, batch, lease_seconds=900):
        return self.leased[:batch]

    def finish(self, source, aid, state, error=None):
        self.finished[aid] = state

    def release(self, source, ids):
        self.released.extend(ids)
        return len(ids)

    # The latch gate runs before any request; no host is latched here.
    def latched(self, host):
        return None

    def record_refusal(self, host, status, threshold, detail=""):
        return False

    def record_success(self, host):
        pass


def test_fetch_returns_unreached_leases_when_the_deadline_hits(monkeypatch, tmp_path) -> None:
    """The livelock this fixes: a batch that cannot finish inside the Job's
    activeDeadlineSeconds was SIGKILLed holding every lease it took, so the next
    tick found nothing claimable for a full lease window and did nothing."""
    from lake.config import Config
    from pipeline import fetch

    f = LeaseFrontier(100)
    cfg = Config(
        database_url="",
        blob_root=str(tmp_path),
        user_agent="t",
        civitai_api_key=None,
        api_rps=1000,
        cdn_rps=1000,
        keep_prefix_days=90,
        fetch_deadline=0.0,
    )
    # Deadline 0: the loop must stop before the first artifact and hand all 100 back.
    stats = fetch.run(cfg, f, batch=100)
    assert stats["stopped_early"] == "deadline"
    assert stats["processed"] == 0
    assert len(f.released) == 100, "every unreached lease must go back to pending"
    assert stats["released"] == 100


def test_fetch_reports_processed_not_leased(monkeypatch, tmp_path) -> None:
    """crawl_run.items showed the leased count, so a truncated run looked
    identical to a complete one."""
    from lake.config import Config
    from pipeline import fetch

    f = LeaseFrontier(10)
    cfg = Config(
        database_url="",
        blob_root=str(tmp_path),
        user_agent="t",
        civitai_api_key=None,
        api_rps=1000,
        cdn_rps=1000,
        keep_prefix_days=90,
        fetch_deadline=0.0,
    )
    fetch.run(cfg, f, batch=10)
    assert f.last_run["items"] == 0, "items must reflect work done, not work leased"
