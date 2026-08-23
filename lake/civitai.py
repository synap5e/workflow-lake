"""Civitai access layer used by the probes.

Two discovery channels, because the obvious one is gutted:

* `/api/v1/images` — the documented REST API. Fine for enumerating images with
  author/date/stats, but `meta` is **always null** (measured: 0/N across sorts,
  authenticated and not), so it cannot tell you whether an image is ComfyUI.
* `/api/trpc/*` — the site's own API. `image.getInfinite` supports the `tools`
  filter (ComfyUI is tool id 86) and returns a `hasMeta` flag;
  `image.getGenerationData` returns the full generation metadata including the
  ComfyUI graph when one was recorded. Responses are devalue-flattened.

Both are public; nothing here logs in. The API key, when present, is only sent
to civitai.com.
"""

from __future__ import annotations

import json
import os
import pathlib
import urllib.parse
from collections.abc import Iterator
from typing import Any

from .devalue import trpc_result
from .polite import HostBlocked, PoliteClient

REST = "https://civitai.com/api/v1"
TRPC = "https://civitai.com/api/trpc"
TOOL_COMFYUI = 86

# Every host this source touches. The latch gate checks all of them before a run,
# and medina's egress NetworkPolicy is written against this list.
EGRESS_HOSTS = ["civitai.com", "image.civitai.com", "image-b2.civitai.com"]


def api_key() -> str | None:
    """Key from $CIVITAI_API_KEY or the file named by $CIVITAI_API_KEY_FILE."""
    key = os.environ.get("CIVITAI_API_KEY")
    if key:
        return key.strip()
    path = os.environ.get("CIVITAI_API_KEY_FILE")
    if path and pathlib.Path(path).exists():
        return pathlib.Path(path).read_text().strip()
    return None


def auth_headers() -> dict[str, str]:
    key = api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


def trpc(client: PoliteClient, route: str, payload: dict[str, Any]) -> Any:
    url = f"{TRPC}/{route}?input=" + urllib.parse.quote(json.dumps(payload))
    resp = client.get(url, headers={**auth_headers(), "Accept": "application/json"})
    resp.raise_for_status()
    return trpc_result(resp.content)


def rest_images(
    client: PoliteClient,
    *,
    limit: int,
    sort: str = "Newest",
    period: str = "AllTime",
    nsfw: str = "None",
) -> Iterator[dict]:
    """Enumerate images through the documented REST API."""
    cursor: str | None = None
    yielded = 0
    while yielded < limit:
        params = {"limit": "100", "sort": sort, "period": period, "nsfw": nsfw}
        if cursor:
            params["cursor"] = cursor
        resp = client.get(f"{REST}/images", params=params, headers=auth_headers())
        if resp.status_code != 200:
            return
        body = resp.json()
        items = body.get("items") or []
        if not items:
            return
        for item in items:
            yield item
            yielded += 1
            if yielded >= limit:
                return
        cursor = (body.get("metadata") or {}).get("nextCursor")
        if not cursor:
            return


def trpc_images(
    client: PoliteClient,
    *,
    limit: int,
    tools: list[int] | None = None,
    sort: str = "Newest",
    period: str = "AllTime",
    browsing_level: int = 1,
) -> Iterator[dict]:
    """Enumerate images through the site API, optionally filtered by tool."""
    cursor: Any = None
    yielded = 0
    while yielded < limit:
        req: dict[str, Any] = {
            "period": period,
            "sort": sort,
            "browsingLevel": browsing_level,
            "limit": min(100, limit - yielded),
        }
        if tools:
            req["tools"] = tools
        if cursor:
            req["cursor"] = cursor
        data = trpc(client, "image.getInfinite", {"json": req})
        items = (data or {}).get("items") or []
        if not items:
            return
        for item in items:
            yield item
            yielded += 1
            if yielded >= limit:
                return
        cursor = (data or {}).get("nextCursor")
        if not cursor:
            return


def generation_data(client: PoliteClient, image_id: int) -> dict | None:
    """Civitai's own record of how an image was made (its 'API-only' channel)."""
    try:
        return trpc(client, "image.getGenerationData", {"json": {"id": image_id}})
    except HostBlocked:
        raise
    except Exception as exc:
        if _is_auth_failure(exc):
            raise AuthRequired("image.getGenerationData refused (HTTP 401/403)") from exc
        return None


def image_url(item: dict) -> str | None:
    """Absolute original-bytes URL for a tRPC or REST image item.

    REST items carry a full `url` already pointing at `original=true`. tRPC items
    carry a bare UUID in `url`, which the CDN serves under a transform segment.
    """
    url = item.get("url")
    if not url:
        return None
    if url.startswith("http"):
        return url
    name = item.get("name") or f"{url}.png"
    return f"https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/{url}/original=true/{name}"


# --- production discovery ---------------------------------------------------
#
# Paging `image.getInfinite` backwards from the tip caps at ~4,400 items: after
# 44 pages the cursor stops advancing and the endpoint returns a fixed jumbled
# fallback set. So the tip is paged (shallowly, until it meets a known id) and
# history is reached one partition at a time instead.

TIP_PAGE_CAP = 40  # stay well inside the ~44-page cursor cap


class AuthRequired(RuntimeError):
    """The source refused us for lack of (or with a bad) credential.

    Distinct from "this artifact is gone", and it must stay distinct: swallowing
    a 401 as a 404 would march the whole queue to `skipped: gone` and report a
    clean run, which is the same exit-0-while-doing-nothing shape that has
    already bitten this project twice.
    """


def _is_auth_failure(exc: Exception) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403)


def image_get(client: PoliteClient, image_id: int | str) -> dict | None:
    """Single image by id. Returns None only when the id is genuinely gone."""
    try:
        data = trpc(client, "image.get", {"json": {"id": int(image_id)}})
    except HostBlocked:
        # The latch fired. It must reach the job, not be mistaken for a 404.
        raise
    except Exception as exc:
        if _is_auth_failure(exc):
            raise AuthRequired(
                "image.get refused (HTTP 401/403). Civitai's tRPC routes require "
                "a token; set CIVITAI_API_KEY_FILE."
            ) from exc
        return None
    return data if isinstance(data, dict) and data.get("id") else None


def unwrap_date(value: Any) -> str | None:
    """tRPC dates arrive devalue-tagged as ['Date', iso]."""
    if isinstance(value, list) and len(value) == 2 and value[0] == "Date":
        return value[1]
    return value if isinstance(value, str) else None


def page_images(
    client: PoliteClient,
    *,
    cursor: Any = None,
    tools: list[int] | None = None,
    sort: str = "Newest",
    period: str = "AllTime",
    browsing_level: int = 31,
    limit: int = 100,
    **partition: Any,
) -> tuple[list[dict], Any]:
    """One page of `image.getInfinite`, returning (items, next_cursor).

    `partition` takes `username`, `userId`, `postId` or `modelVersionId`. Paging
    *within* a partition is clean, which is the only way to reach history.
    """
    req: dict[str, Any] = {
        "period": period,
        "sort": sort,
        "browsingLevel": browsing_level,
        "limit": limit,
    }
    if tools:
        req["tools"] = tools
    if cursor:
        req["cursor"] = cursor
    req.update({k: v for k, v in partition.items() if v is not None})
    data = trpc(client, "image.getInfinite", {"json": req})
    return (data or {}).get("items") or [], (data or {}).get("nextCursor")


def sidecar(item: dict) -> dict[str, Any]:
    """The lake's per-artifact metadata, projected out of a listing item."""
    user = item.get("user") or {}
    stats = item.get("stats") or {}
    return {
        "author": user.get("username"),
        "author_id": item.get("userId"),
        "title": "",
        "description": "",
        "tags": [str(t) for t in (item.get("tagIds") or [])],
        "published_at": unwrap_date(item.get("publishedAt")) or unwrap_date(item.get("createdAt")),
        "stats": {k: int(v) for k, v in stats.items() if isinstance(v, (int, float))},
    }


def prefilter(item: dict) -> dict[str, Any]:
    """Why we would queue this artifact, recorded so the policy can be re-scored.

    `hasMeta OR toolIds ∋ ComfyUI` keeps 99.2% of workflows while fetching 76.8%
    of artifacts; `toolIds` alone keeps 87.4% while fetching 31.8%.
    """
    tools = item.get("toolIds") or []
    return {
        "has_meta": bool(item.get("hasMeta")),
        "comfy_tagged": TOOL_COMFYUI in tools,
        "tool_ids": tools,
        "type": item.get("type"),
    }


def wanted(item: dict, *, mode: str) -> bool:
    if mode == "all":
        return True
    flags = prefilter(item)
    if mode == "comfy":
        return flags["comfy_tagged"]
    return flags["comfy_tagged"] or flags["has_meta"]  # "complete"
