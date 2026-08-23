"""The capture job: lease a batch, fetch it, write blobs and a manifest.

This is the only job that spends bandwidth and the only one whose output cannot
be reconstructed later, so it is deliberately dull. It reads work from Postgres,
writes bytes to the bucket, and marks rows done. It does not derive anything, it
does not talk to ClickHouse, and it holds no state of its own between runs.

Two rate budgets, because they are different hosts with different tolerances:
the API host (`civitai.com`) and the image CDN.
"""

from __future__ import annotations

import signal
import sys
import time
from typing import Any

from lake import civitai
from lake.capture import capture_record, fetch_bytes, should_try_api
from lake.config import Config
from lake.db import Frontier
from lake.latch import PostgresLatch, gate
from lake.polite import HostBlocked, PoliteClient
from lake.storage import BlobWriter, ManifestWriter, open_store

SOURCE = "civitai"


def _api_channel(client: PoliteClient, artifact_id: str) -> tuple[dict, dict]:
    """Civitai's own generation record. Returns (payloads, raw)."""
    gen = civitai.generation_data(client, int(artifact_id)) or {}
    meta = gen.get("meta") if isinstance(gen.get("meta"), dict) else {}
    comfy = (meta or {}).get("comfy")
    if isinstance(comfy, str):
        import json

        try:
            comfy = json.loads(comfy)
        except Exception:
            comfy = None
    payloads: dict[str, Any] = {}
    if isinstance(comfy, dict):
        for kind in ("workflow", "prompt"):
            if comfy.get(kind) is not None:
                payloads[kind] = comfy[kind]
    return payloads, gen


def run(
    cfg: Config,
    frontier: Frontier,
    *,
    batch: int = 200,
    keep_prefix: bool = True,
    use_api: bool = True,
    deadline_seconds: float | None = None,
) -> dict:
    """Capture a leased batch, bounded by wall clock as well as by size.

    The size bound alone is not enough. Throughput depends on the source's rate
    limit, not on us, so any fixed batch is a guess — and a batch that cannot
    finish inside the job's `activeDeadlineSeconds` gets SIGKILLed holding every
    lease it took, which starves the next tick for a full lease window. The
    deadline here is the real limit; `--batch` is only a ceiling.
    """
    import json

    # Refuse to start if a source we need has told us to stop. Checked before
    # anything is leased, so a latched run costs one query and no requests.
    gate(frontier, civitai.EGRESS_HOSTS)

    run_id = frontier.start_run("fetch", SOURCE)
    store = open_store(cfg.blob_root)
    manifest = ManifestWriter(store, SOURCE, run_id)
    blobs = BlobWriter(store, frontier, cfg.keep_prefix_days)

    latch = PostgresLatch(frontier)
    api = PoliteClient(rps=cfg.api_rps, user_agent=cfg.user_agent, latch=latch)
    cdn = PoliteClient(rps=cfg.cdn_rps, user_agent=cfg.user_agent, latch=latch)

    items = frontier.claim(SOURCE, batch)
    print(f"leased {len(items)} artifacts (run {run_id})", file=sys.stderr)

    started = time.time()
    deadline = deadline_seconds if deadline_seconds is not None else cfg.fetch_deadline
    workflows = 0
    failed = 0
    processed: set[str] = set()
    stopped_early = None

    # SIGTERM is what k8s sends before SIGKILL. Finish the artifact in hand,
    # then stop and hand the rest back.
    terminating = False

    def on_term(_sig, _frame):
        nonlocal terminating
        terminating = True
        print("SIGTERM: finishing current artifact, then releasing leases", file=sys.stderr)

    signal.signal(signal.SIGTERM, on_term)

    try:
        for index, row in enumerate(items):
            elapsed = time.time() - started
            if terminating or elapsed > deadline:
                stopped_early = "sigterm" if terminating else "deadline"
                break
            if index and index % 50 == 0:
                rate = index / max(elapsed, 0.001)
                print(
                    f"  {index}/{len(items)} in {elapsed:.0f}s ({rate:.2f}/s, "
                    f"{workflows} workflows)",
                    file=sys.stderr,
                )
            artifact_id = row["artifact_id"]
            prefilter = row.get("prefilter") or {}
            parent_url = f"https://civitai.com/images/{artifact_id}"
            try:
                # The listing does not carry the CDN path, so the artifact's own
                # record is what resolves it. It also refreshes the sidecar, which
                # may have changed since discovery.
                record = civitai.image_get(api, artifact_id)
                if record is None:
                    frontier.finish(SOURCE, artifact_id, "skipped", "gone (404)")
                    continue
                url = civitai.image_url(record)
                side = civitai.sidecar(record)

                result = fetch_bytes(cdn, url, keep_prefix=keep_prefix)
                prefix_sha = None
                if keep_prefix and result.prefix:
                    prefix_sha, _, _ = blobs.put(result.prefix, "prefix")

                wrote_any = False
                for kind in ("workflow", "prompt"):
                    payload = getattr(result, kind)
                    if payload is None:
                        continue
                    sha, _key, size = blobs.put(payload.encode(), kind)
                    manifest.write(
                        capture_record(
                            source=SOURCE,
                            artifact_id=artifact_id,
                            channel=result.channel,
                            run_id=run_id,
                            parent_url=parent_url,
                            fetch_url=url,
                            sidecar=side,
                            source_meta=record,
                            result=result,
                            payload_kind=kind,
                            payload_sha=sha,
                            payload_bytes=size,
                            prefix_sha=prefix_sha,
                        )
                    )
                    wrote_any = True

                if use_api and should_try_api(result, prefilter):
                    payloads, gen = _api_channel(api, artifact_id)
                    for kind, payload in payloads.items():
                        blob = json.dumps(payload).encode()
                        sha, _key, size = blobs.put(blob, kind)
                        manifest.write(
                            capture_record(
                                source=SOURCE,
                                artifact_id=artifact_id,
                                channel="api",
                                run_id=run_id,
                                parent_url=parent_url,
                                fetch_url="image.getGenerationData",
                                sidecar=side,
                                source_meta=gen,
                                result=None,
                                payload_kind=kind,
                                payload_sha=sha,
                                payload_bytes=size,
                            )
                        )
                        wrote_any = True

                if wrote_any:
                    workflows += 1
                else:
                    # A negative is a real, useful result: it says this artifact
                    # holds nothing, so nobody fetches it again.
                    manifest.write(
                        capture_record(
                            source=SOURCE,
                            artifact_id=artifact_id,
                            channel=result.channel,
                            run_id=run_id,
                            parent_url=parent_url,
                            fetch_url=url,
                            sidecar=side,
                            source_meta=record,
                            result=result,
                            payload_kind="none",
                            payload_sha=None,
                            payload_bytes=0,
                            prefix_sha=prefix_sha,
                        )
                    )
                frontier.finish(SOURCE, artifact_id, "done")
                processed.add(artifact_id)
            except (civitai.AuthRequired, HostBlocked):
                processed.add(artifact_id)
                # Not this artifact's problem — every remaining one will fail the
                # same way. Return the lease and stop rather than burning the
                # queue into `failed` one row at a time.
                frontier.finish(SOURCE, artifact_id, "pending")
                raise
            except Exception as exc:
                failed += 1
                frontier.finish(SOURCE, artifact_id, "failed", f"{type(exc).__name__}: {exc}")
                processed.add(artifact_id)
    finally:
        # Anything leased but not reached goes straight back to pending, so the
        # next tick picks it up in seconds rather than after the lease expires.
        released = frontier.release(
            SOURCE, [r["artifact_id"] for r in items if r["artifact_id"] not in processed]
        )
        key = manifest.commit()
        api.close()
        cdn.close()

    stats = {
        "items": len(items),
        "processed": len(processed),
        "released": released,
        "stopped_early": stopped_early,
        "workflows": workflows,
        "failed": failed,
        "records": manifest.count,
        "blobs_written": blobs.written,
        "blobs_deduped": blobs.deduped,
        "requests": api.stats.requests + cdn.stats.requests,
        "bytes_down": api.stats.bytes_down + cdn.stats.bytes_down,
        "manifest_key": key,
        "seconds": round(time.time() - started, 1),
    }
    frontier.finish_run(
        run_id,
        items=stats["processed"],
        workflows=stats["workflows"],
        requests=stats["requests"],
        bytes_down=stats["bytes_down"],
        manifest_key=key,
    )
    return stats
