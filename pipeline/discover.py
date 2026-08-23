"""Discovery: deciding what to fetch, without fetching it.

Two streams, because Civitai's cursor caps at ~4,400 items and they therefore
need completely different shapes:

* **tip** — page `sort=Newest` until a page holds nothing we do not already
  have, then stop. Bounded by how much was published since the last run, so a
  15-minute cadence never comes near the cursor cap.
* **backlog** — drain one partition at a time. Partitions come from the tip:
  every artifact carries a `userId`, so the tip crawl accumulates the partition
  list and the backlog crawl works through it. History is a breadth-first
  expansion out of the tip rather than a walk backwards through it.

Both write into `crawl_queue` and nothing else. Fetching is a separate job so
that a discovery bug never costs bandwidth.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import sys

from lake import civitai
from lake.config import Config
from lake.db import Frontier, QueueItem
from lake.latch import PostgresLatch, gate
from lake.polite import PoliteClient

SOURCE = "civitai"
TIP_PRIORITY = 100
BACKLOG_PRIORITY = 0

# How many consecutive pages must add nothing before the tip is considered
# caught up. See the loop in `discover_tip` for why one is not enough.
DRY_PAGES_TO_STOP = 2


def _published(items: list[dict]) -> list[dt.datetime]:
    """publishedAt for every item that has a parseable one."""
    out = []
    for item in items:
        raw = item.get("publishedAt") or item.get("createdAt")
        if raw:
            with contextlib.suppress(ValueError):
                out.append(dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
    return out


def _to_items(rows: list[dict], frontier: Frontier, one_per_post: bool) -> list[QueueItem]:
    """Turn listing items into queue rows, applying the one-per-post policy.

    91% of multi-image posts share a single workflow topology, so the second
    artifact of a post is nearly always a duplicate graph. Dropping it costs 8%
    of distinct topologies and 2.5x fewer fetches.
    """
    seen_posts: set[str] = set()
    if one_per_post:
        post_ids = [str(r["postId"]) for r in rows if r.get("postId")]
        seen_posts = frontier.posts_already_captured(SOURCE, post_ids)

    out: list[QueueItem] = []
    for row in rows:
        post_id = str(row["postId"]) if row.get("postId") else None
        if one_per_post and post_id:
            if post_id in seen_posts:
                continue
            seen_posts.add(post_id)
        side = civitai.sidecar(row)
        published = side.get("published_at")
        out.append(
            QueueItem(
                source=SOURCE,
                artifact_id=str(row["id"]),
                post_id=post_id,
                author_id=str(row.get("userId")) if row.get("userId") else None,
                published_at=dt.datetime.fromisoformat(published.replace("Z", "+00:00"))
                if published
                else None,
                prefilter=civitai.prefilter(row),
            )
        )
    return out


def discover_tip(
    cfg: Config,
    frontier: Frontier,
    *,
    mode: str = "complete",
    max_pages: int = civitai.TIP_PAGE_CAP,
    one_per_post: bool = True,
) -> dict:
    """Page Newest until we reach known ground."""
    gate(frontier, civitai.EGRESS_HOSTS)
    # Recorded like fetch: `crawl_run` is what a "no successful run in N hours"
    # alert reads, and discovery silently absent from it would make a stalled
    # crawler look healthy.
    run_id = frontier.start_run("discover-tip", SOURCE)
    client = PoliteClient(rps=cfg.api_rps, user_agent=cfg.user_agent, latch=PostgresLatch(frontier))
    state = frontier.get_cursor(SOURCE, "tip") or {}

    cursor = None
    queued = partitions = pages = dry_pages = 0
    newest_id = int(state.get("frontier_id") or 0)
    newest_published: dt.datetime | None = None
    caught_up = False

    try:
        for _ in range(max_pages):
            items, cursor = civitai.page_images(client, cursor=cursor)
            if not items:
                break
            pages += 1
            newest_id = max(newest_id, max(int(i["id"]) for i in items))
            newest_published = max(
                [newest_published, *_published(items)] if newest_published else _published(items),
                default=None,
            )

            # Partitions come from every artifact seen, wanted or not: a user who
            # posts anything is a user whose whole history is reachable.
            partitions += frontier.add_partitions(
                SOURCE,
                "user",
                sorted({str(i["userId"]) for i in items if i.get("userId")}),
            )
            keep = [i for i in items if civitai.wanted(i, mode=mode)]
            # Known ground is measured by what the INSERT actually inserted, not
            # by comparing ids against a watermark. `enqueue` is ON CONFLICT DO
            # NOTHING and returns the number of genuinely new rows, so a page
            # that adds nothing is a page we have already seen — whatever order
            # the source chose to return it in.
            new_rows = frontier.enqueue(
                _to_items(keep, frontier, one_per_post), priority=TIP_PRIORITY
            )
            queued += new_rows

            # One barren page is not proof: `wanted` can reject a whole page of
            # genuinely new artifacts, and stopping there would leave anything
            # behind it unreachable. Two in a row means we are past the new work.
            dry_pages = dry_pages + 1 if new_rows == 0 else 0
            if dry_pages >= DRY_PAGES_TO_STOP:
                caught_up = True
                break
            if not cursor:
                break
    finally:
        client.close()

    frontier.set_cursor(SOURCE, "tip", None, str(newest_id) if newest_id else None)
    frontier.finish_run(
        run_id, items=queued, requests=client.stats.requests, bytes_down=client.stats.bytes_down
    )
    return {
        "pages": pages,
        "queued": queued,
        "new_partitions": partitions,
        "frontier_id": newest_id,
        "newest_published": newest_published.isoformat() if newest_published else None,
        "caught_up": caught_up,
        "requests": client.stats.requests,
    }


def discover_backlog(
    cfg: Config,
    frontier: Frontier,
    *,
    mode: str = "comfy",
    partitions: int = 5,
    max_pages_per_partition: int = 40,
    one_per_post: bool = True,
) -> dict:
    """Drain un-swept partitions, one at a time, resuming where each left off."""
    gate(frontier, civitai.EGRESS_HOSTS)
    run_id = frontier.start_run("discover-backlog", SOURCE)
    client = PoliteClient(rps=cfg.api_rps, user_agent=cfg.user_agent, latch=PostgresLatch(frontier))
    queued = drained = exhausted = 0
    try:
        for _ in range(partitions):
            part = frontier.claim_partition(SOURCE, "user")
            if part is None:
                break
            drained += 1
            cursor = part["cursor"]
            pages = part["pages_read"]
            images = part["images_seen"]
            done = False
            error = None
            try:
                for _page in range(max_pages_per_partition):
                    items, cursor = civitai.page_images(
                        client, cursor=cursor, userId=int(part["key"])
                    )
                    if not items:
                        done = True
                        break
                    pages += 1
                    images += len(items)
                    keep = [i for i in items if civitai.wanted(i, mode=mode)]
                    queued += frontier.enqueue(
                        _to_items(keep, frontier, one_per_post), priority=BACKLOG_PRIORITY
                    )
                    if not cursor:
                        done = True
                        break
            except Exception as exc:  # one bad partition must not kill the run
                error = f"{type(exc).__name__}: {exc}"
                print(f"  partition {part['key']}: {error}", file=sys.stderr)
            exhausted += 1 if done else 0
            frontier.update_partition(
                SOURCE,
                "user",
                part["key"],
                cursor=cursor,
                exhausted=done,
                pages_read=pages,
                images_seen=images,
                error=error,
            )
    finally:
        client.close()

    frontier.finish_run(
        run_id, items=queued, requests=client.stats.requests, bytes_down=client.stats.bytes_down
    )
    return {
        "partitions_drained": drained,
        "partitions_exhausted": exhausted,
        "queued": queued,
        "requests": client.stats.requests,
    }
