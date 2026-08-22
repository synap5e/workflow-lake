"""Fetching one artifact across its channels, and the capture records that fall out.

This is the production form of `probes/civitai_images.py`'s `probe_bytes`, with
the policies that the experiments settled baked in:

* **Container-dispatched ranging.** A 128 KB head for PNG/JPEG; a 128 KB *tail*
  for video, because ffmpeg writes `moov` after `mdat` and 47 of 49 sampled video
  workflows were reachable only from the end of the file. WebP also keeps its
  EXIF after the image data.
* **Continuation escalation.** When the head runs out mid-chunk, fetch the *next*
  slice and append rather than re-fetching from zero. The probes restarted, which
  on small files cost more than the whole file.
* **Bytes before API.** The API channel is one request against the rate-limited
  host, and `hasMeta` is a perfect necessary condition for it holding anything.
  Gating on "bytes failed AND hasMeta" cut API calls by up to 81% while losing
  no workflows at all.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from .config import CRAWLER_VERSION, PARSER_VERSION
from .polite import PoliteClient, total_size_from
from .wfmeta import parse

HEAD_WINDOW = 128 * 1024
MAX_HEAD = 512 * 1024
TAIL_WINDOW = 128 * 1024


@dataclass
class ChannelResult:
    channel: str
    http_status: int | None = None
    container: str | None = None
    content_type: str | None = None
    total_size: int | None = None
    bytes_fetched: int = 0
    needed_bytes: int | None = None
    windows: list[int] = field(default_factory=list)
    from_tail: bool = False
    workflow: str | None = None
    prompt: str | None = None
    prefix: bytes | None = None
    error: str | None = None

    @property
    def has_payload(self) -> bool:
        return self.workflow is not None or self.prompt is not None


def fetch_bytes(client: PoliteClient, url: str, *, keep_prefix: bool = True) -> ChannelResult:
    """Ranged fetch and parse of one artifact's served bytes."""
    out = ChannelResult(channel="bytes_head")
    data = b""
    total: int | None = None
    start = 0
    end = HEAD_WINDOW

    while True:
        resp = client.get_range(url, start, end - 1)
        out.http_status = resp.status_code
        if resp.status_code not in (200, 206):
            out.error = f"http {resp.status_code}"
            return out
        chunk = resp.content if resp.status_code == 206 else resp.content[:end]
        data += chunk
        out.bytes_fetched += len(resp.content)
        out.windows.append(len(chunk))
        total = total_size_from(resp) or total
        out.total_size = total
        out.content_type = resp.headers.get("content-type")

        head = parse(data, scan_past_image_data=False)
        out.container = head.fmt
        out.needed_bytes = head.needed_bytes
        out.workflow, out.prompt = head.workflow, head.prompt

        # A second exhaustive pass catches anything sitting past the pixel data
        # that happened to land inside the window.
        full = parse(data, scan_past_image_data=True)
        for kind in ("workflow", "prompt"):
            if getattr(full, kind) and getattr(out, kind) is None:
                setattr(out, kind, getattr(full, kind))

        if head.needs_tail:
            break
        more = total is None or len(data) < total
        if not (head.truncated and more and not out.workflow) or end >= MAX_HEAD:
            break
        # Continuation, not a restart.
        start, end = end, min(end * 4, MAX_HEAD)

    # Video parks its metadata at the end of the file; so does WebP.
    wants_tail = (out.container or "").startswith("isobmff") or (
        out.container == "webp" and not out.workflow
    )
    if wants_tail and not out.workflow and total and total > TAIL_WINDOW:
        tail = client.get_tail(url, TAIL_WINDOW)
        if tail.status_code == 206:
            out.bytes_fetched += len(tail.content)
            parsed = parse(tail.content)
            for kind in ("workflow", "prompt"):
                if getattr(parsed, kind) and getattr(out, kind) is None:
                    setattr(out, kind, getattr(parsed, kind))
                    out.from_tail = True
            if out.from_tail:
                out.channel = "bytes_tail"
            if keep_prefix:
                out.prefix = tail.content
    if keep_prefix and out.prefix is None:
        out.prefix = data
    return out


def should_try_api(bytes_result: ChannelResult, prefilter: dict[str, Any]) -> bool:
    """Spend an API request only when it can plausibly pay.

    Measured: gating on `hasMeta` cut API calls by up to 81% and lost nothing,
    because Civitai sets the flag exactly when it recorded generation metadata.
    """
    if bytes_result.workflow is not None:
        return False
    return bool(prefilter.get("has_meta"))


def capture_record(
    *,
    source: str,
    artifact_id: str,
    channel: str,
    run_id: str,
    parent_url: str,
    fetch_url: str,
    sidecar: dict[str, Any],
    source_meta: dict[str, Any],
    result: ChannelResult | None,
    payload_kind: str,
    payload_sha: str | None,
    payload_bytes: int,
    prefix_sha: str | None = None,
) -> dict[str, Any]:
    """One line of the manifest: the row `raw_artifacts` is loaded from."""
    return {
        "capture_id": f"{source}:{artifact_id}:{channel}:{payload_kind}",
        "crawled_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "crawler_version": CRAWLER_VERSION,
        "parser_version": PARSER_VERSION,
        "run_id": run_id,
        "source": source,
        "channel": channel,
        "source_artifact_id": str(artifact_id),
        "parent_url": parent_url,
        "fetch_url": fetch_url,
        "author": sidecar.get("author") or "",
        "author_id": str(sidecar.get("author_id") or ""),
        "title": sidecar.get("title") or "",
        "description": sidecar.get("description") or "",
        "tags": sidecar.get("tags") or [],
        "published_at": sidecar.get("published_at"),
        "stats": sidecar.get("stats") or {},
        # The untouched source payload. Everything above is a projection of it,
        # and this is the only part of the pipeline that cannot be redone later.
        "source_meta": source_meta,
        "http_status": (result.http_status if result else 200) or 0,
        "content_type": (result.content_type if result else "") or "",
        "container": (result.container if result else "") or "",
        "total_size": result.total_size if result else None,
        "bytes_fetched": result.bytes_fetched if result else 0,
        "needed_bytes": result.needed_bytes if result else None,
        "windows": result.windows if result else [],
        "from_tail": bool(result.from_tail) if result else False,
        "payload_kind": payload_kind,
        "payload_sha256": payload_sha,
        "payload_bytes": payload_bytes,
        "prefix_sha256": prefix_sha,
    }
