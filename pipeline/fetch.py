"""The capture job: lease a batch, fetch it, write blobs and a manifest.

This is the only job that spends bandwidth and the only one whose output cannot
be reconstructed later, so it is deliberately dull. It reads work from Postgres,
writes bytes to the bucket, and marks rows done. It does not derive anything, it
does not talk to ClickHouse, and it holds no state of its own between runs.

Two rate budgets, because they are different hosts with different tolerances:
the API host (`civitai.com`) and the image CDN.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from lake import civitai
from lake.capture import capture_record, fetch_bytes, should_try_api
from lake.config import Config
from lake.db import Frontier
from lake.latch import PostgresLatch, gate
from lake.polite import PoliteClient
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
) -> dict:
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
    workflows = 0
    failed = 0

    try:
        for row in items:
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
            except Exception as exc:
                failed += 1
                frontier.finish(SOURCE, artifact_id, "failed", f"{type(exc).__name__}: {exc}")
    finally:
        key = manifest.commit()
        api.close()
        cdn.close()

    stats = {
        "items": len(items),
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
        items=stats["items"],
        workflows=stats["workflows"],
        requests=stats["requests"],
        bytes_down=stats["bytes_down"],
        manifest_key=key,
    )
    return stats
