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


class TipFrontier:
    """Enough Frontier to exercise tip discovery, with real ON CONFLICT semantics."""

    def __init__(self, known: set[str] | None = None) -> None:
        self.known = set(known or ())
        self.cursor_state: dict | None = None
        self.partitions: set[str] = set()

    def start_run(self, job, source):
        return "run-1"

    def finish_run(self, run_id, **kw):
        self.last_run = kw

    def get_cursor(self, source, stream):
        return self.cursor_state

    def set_cursor(self, source, stream, cursor, frontier_id):
        self.cursor_state = {"cursor": cursor, "frontier_id": frontier_id}

    def add_partitions(self, source, kind, ids):
        new = set(ids) - self.partitions
        self.partitions |= new
        return len(new)

    def posts_already_captured(self, source, post_ids):
        return set()

    def enqueue(self, items, *, priority):
        """Mirrors ON CONFLICT DO NOTHING: only genuinely new ids count."""
        new = [i for i in items if i.artifact_id not in self.known]
        self.known.update(i.artifact_id for i in new)
        return len(new)

    def latched(self, host):
        return None

    def record_refusal(self, host, status, threshold, detail=""):
        return False

    def record_success(self, host):
        pass


def _listing_item(image_id: int, published: str, *, comfy: bool = True) -> dict:
    return {
        "id": image_id,
        "postId": image_id * 10,
        "userId": 7,
        "publishedAt": published,
        "toolIds": [86] if comfy else [],
        "hasMeta": False,
        "type": "image",
        "url": f"https://image.civitai.com/{image_id}/x.png",
    }


def test_tip_advances_when_publish_order_disagrees_with_id_order(monkeypatch) -> None:
    """The stall this fixes.

    Civitai's `sort=Newest` orders by publishedAt, but ids are assigned at
    UPLOAD. Someone uploads privately and publishes days later, so a
    freshly-published image can carry an id far below one we already hold — a
    single observed page spanned 4.7M ids. Discovery used `max(id)` as its
    watermark, so once a high id landed, every subsequently-published artifact
    looked old, `fresh` was empty forever and the queue drained to zero pending
    and stayed there.
    """
    from lake.config import Config
    from pipeline import discover

    # We already hold 500, the highest id. Everything published SINCE is lower.
    frontier = TipFrontier(known={"500"})
    frontier.set_cursor("civitai", "tip", None, "500")
    pages = [
        [_listing_item(140, "2026-08-23T05:38:00Z"), _listing_item(500, "2026-08-23T05:31:00Z")],
        [_listing_item(141, "2026-08-23T05:20:00Z")],
        [],
    ]
    monkeypatch.setattr(
        discover.civitai,
        "page_images",
        lambda client, cursor=None, **kw: (pages[cursor or 0], (cursor or 0) + 1),
    )
    cfg = Config(
        database_url="",
        blob_root="",
        user_agent="t",
        civitai_api_key=None,
        api_rps=1000,
        cdn_rps=1000,
        keep_prefix_days=90,
        fetch_deadline=600.0,
    )

    out = discover.discover_tip(cfg, frontier, max_pages=5)

    assert out["queued"] == 2, "ids below the watermark are still new work"
    assert {"140", "141"} <= frontier.known
    assert out["newest_published"] is not None


def test_tip_stops_once_pages_stop_adding_anything(monkeypatch) -> None:
    """Known ground is 'this page inserted nothing', twice — not an id compare.

    One barren page is not enough: `wanted` can reject a whole page of genuinely
    new artifacts, and stopping there would strand everything behind it.
    """
    from lake.config import Config
    from pipeline import discover

    frontier = TipFrontier(known={"1", "2", "3"})
    pages = [
        [_listing_item(1, "2026-08-23T05:00:00Z")],  # dry 1
        [_listing_item(9, "2026-08-23T04:59:00Z")],  # resets the counter
        [_listing_item(2, "2026-08-23T04:58:00Z")],  # dry 1
        [_listing_item(3, "2026-08-23T04:57:00Z")],  # dry 2 -> stop
        [_listing_item(99, "2026-08-23T04:56:00Z")],  # must never be reached
    ]
    seen_pages = []

    def _page(client, cursor=None, **kw):
        idx = cursor or 0
        seen_pages.append(idx)
        return pages[idx], idx + 1

    monkeypatch.setattr(discover.civitai, "page_images", _page)
    cfg = Config(
        database_url="",
        blob_root="",
        user_agent="t",
        civitai_api_key=None,
        api_rps=1000,
        cdn_rps=1000,
        keep_prefix_days=90,
        fetch_deadline=600.0,
    )

    out = discover.discover_tip(cfg, frontier, max_pages=10)

    assert out["caught_up"] is True
    assert out["queued"] == 1, "only id 9 was new"
    assert "99" not in frontier.known
    assert len(seen_pages) == 4, "stopped after two consecutive barren pages"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://image.civitai.com/x/y/original=true/a.png", True),
        ("https://image.civitai.com/x/y/original=true/a.webp", True),
        ("https://image.civitai.com/x/y/original=true/a.mp4", True),
        ("https://image.civitai.com/x/y/original=true/A.PNG", True),
        # Query strings must not defeat the extension check.
        ("https://image.civitai.com/x/y/a.jpeg?width=450", False),
    ],
)
def test_only_barren_containers_are_skipped(url, expected) -> None:
    from lake import civitai

    assert civitai.bytes_worth_fetching(url, sample_rate=0.0) is expected


def test_a_stable_sample_of_barren_containers_is_still_fetched() -> None:
    """`never` is a claim with an expiry date.

    JPEG carried a workflow 0 times in 551 artifacts across two independent
    samples, which justifies skipping it — but not forever and not silently. A
    fixed slice keeps being fetched so a change at the source shows up as data
    rather than as an assumption nobody revisits. Keyed on the URL, so a
    re-run makes the same decision instead of drifting per attempt.
    """
    from lake import civitai

    urls = [f"https://image.civitai.com/x/{i}/original=true/{i}.jpeg" for i in range(4000)]
    sampled = [u for u in urls if civitai.bytes_worth_fetching(u)]

    assert 0.01 < len(sampled) / len(urls) < 0.035, "roughly 2%"
    assert all(civitai.bytes_worth_fetching(u) for u in sampled), "decision must be stable"


def test_barren_artifacts_are_finished_not_left_leased(monkeypatch, tmp_path) -> None:
    """A skip is still a decision: the row must leave the queue.

    Skipping by `continue` without finishing would have left every JPEG leased
    until its lease expired, then re-leased it forever — busier than the
    behaviour it replaced.
    """
    from lake import civitai
    from lake.config import Config
    from pipeline import fetch

    f = LeaseFrontier(3)
    monkeypatch.setattr(
        civitai, "image_get", lambda client, aid: {"url": f"u{aid}", "name": "a.jpeg"}
    )
    monkeypatch.setattr(
        civitai,
        "image_url",
        lambda rec: f"https://image.civitai.com/x/{rec['url']}/original=true/a.jpeg",
    )
    monkeypatch.setattr(civitai, "bytes_worth_fetching", lambda url, **kw: False)
    cfg = Config(
        database_url="",
        blob_root=str(tmp_path),
        user_agent="t",
        civitai_api_key=None,
        api_rps=1000,
        cdn_rps=1000,
        keep_prefix_days=90,
        fetch_deadline=600.0,
    )

    stats = fetch.run(cfg, f, batch=3)

    assert stats["barren_skipped"] == 3
    assert stats["processed"] == 3
    assert f.released == [], "a skipped row is finished, not handed back"
    assert set(f.finished.values()) == {"skipped"}


def test_every_ingested_table_dedupes_on_reinsert() -> None:
    """The regression that cost 5-6x duplication in production.

    `raw_artifacts`, `derived_workflow_nodes` and `derived_workflow_bindings`
    shipped as plain MergeTree. Ingest re-read every manifest on every tick
    (see the watermark test below), so those three accumulated a copy of every
    row per run while the two ReplacingMergeTree tables stayed correct.

    The old e2e asserted "re-ingest is idempotent" against `derived_workflows`
    alone — the table that was already safe — so it passed throughout.
    """
    from pipeline.ingest import TABLES, expected_engines

    engines = expected_engines()
    plain = [t for t in TABLES if engines.get(t) != "ReplacingMergeTree"]
    assert plain == [], (
        f"{plain} would duplicate rows if a manifest is ever ingested twice. "
        f"The bucket is the system of record and these tables are rebuildable, "
        f"so loading the same manifest twice has to be a no-op."
    )


def test_bindings_rows_are_distinguishable_per_node() -> None:
    """Without node_id, a binding row has no unique key at all.

    Two nodes of the same class declaring the same input in one workflow
    produced byte-identical rows, so nothing could tell a genuine second
    instance from a re-ingest. Deduplicating would then have silently changed
    the flagship literal-vs-link counts from per-instance to per-distinct-tuple.
    """
    from lake.derive import Reference, derive

    ref = Reference.load()
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}},
    }
    d = derive(graph, ref)
    ckpt = [b for b in d.bindings if b["input"] == "ckpt_name"]

    assert len(ckpt) == 2, "both loader instances must be represented"
    assert {b["node_id"] for b in ckpt} == {"1", "2"}
    keys = {(b["node_id"], b["class_type"], b["input"], b["binding"]) for b in ckpt}
    assert len(keys) == 2, "identical rows cannot be deduplicated without losing a real instance"


class WatermarkFrontier:
    """Records what ingest reads and writes for its watermark."""

    def __init__(self, cursor: str | None = None) -> None:
        self.cursor = cursor
        self.runs: list[str] = []
        self.finished: dict = {}

    def get_cursor(self, source, stream):
        return {"cursor": self.cursor} if self.cursor else None

    def set_cursor(self, source, stream, cursor, frontier_id):
        self.cursor = cursor

    def start_run(self, job, source):
        self.runs.append(job)
        return "run-1"

    def finish_run(self, run_id, **kw):
        self.finished = kw


def test_ingest_resumes_from_its_watermark_instead_of_the_whole_bucket(monkeypatch) -> None:
    """The root cause. `run` computed a watermark, returned it, and nobody stored it.

    `--since` existed but the CronJob never passed it, so every hourly tick
    re-read every manifest ever written. Cost grew with the size of the bucket
    rather than with the work actually available.
    """
    from lake.config import Config
    from pipeline import ingest

    seen: list[str] = []

    class FakeStore:
        def list(self, prefix):
            return ["raw/c/0001.jsonl.gz", "raw/c/0002.jsonl.gz", "raw/c/0003.jsonl.gz"]

    class FakeCH:
        database = "workflow_lake"

        def execute(self, sql):
            return ""

        def query(self, sql):
            return ""

        def insert_jsonl(self, table, lines):
            return len(list(lines))

    monkeypatch.setattr(ingest, "open_store", lambda root: FakeStore())
    monkeypatch.setattr(ingest, "ensure_schema", lambda ch: None)
    monkeypatch.setattr(ingest.Reference, "load", staticmethod(lambda: None))
    monkeypatch.setattr(ingest, "read_manifest", lambda store, key: seen.append(key) or [])
    monkeypatch.setattr(
        ingest, "transform", lambda recs, store, ref: dict.fromkeys(ingest.TABLES, [])
    )
    cfg = Config(
        database_url="",
        blob_root="/x",
        user_agent="t",
        civitai_api_key=None,
        api_rps=1,
        cdn_rps=1,
        keep_prefix_days=90,
        fetch_deadline=600.0,
    )

    # Cold start: nothing consumed yet, so everything is new.
    f = WatermarkFrontier()
    out = ingest.run(cfg, clickhouse=FakeCH(), frontier=f)
    assert out["manifests"] == 3
    assert f.cursor == "raw/c/0003.jsonl.gz", "the watermark must be persisted, not just returned"

    # Second tick: the same bucket, nothing new. This is the run that used to
    # re-ingest all three and multiply every row.
    seen.clear()
    out = ingest.run(cfg, clickhouse=FakeCH(), frontier=f)
    assert out["manifests"] == 0
    assert seen == [], "already-consumed manifests must not be re-read"
    assert f.runs == ["ingest", "ingest"], "ingest must appear in crawl_run like every other job"
