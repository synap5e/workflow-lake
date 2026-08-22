"""Decoder for the `devalue`-flattened payloads Civitai's tRPC routes return.

Civitai's site API (`/api/trpc/*`) is a superjson/devalue transport: the
`result.data` field is a JSON *string* holding a flat array where index 0 is the
root and every integer inside a container is a reference into that same array.
The public REST API (`/api/v1/*`) does not expose the fields we need (see
FINDINGS.md), so the probes speak this instead.

Reference: https://github.com/Rich-Harris/devalue (`stringify`/`parse` format).
"""

from __future__ import annotations

import json
from typing import Any

_HOLE = object()

SPECIALS: dict[int, Any] = {
    -1: None,  # undefined
    -2: _HOLE,  # array hole
    -3: float("nan"),
    -4: float("inf"),
    -5: float("-inf"),
    -6: -0.0,
}


def decode(payload: str | list) -> Any:
    """Decode a devalue payload (JSON string or already-parsed list)."""
    flat = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(flat, list):
        return flat
    memo: dict[int, Any] = {}

    def walk(index: Any) -> Any:
        if not isinstance(index, int) or isinstance(index, bool):
            return index
        if index < 0:
            return SPECIALS.get(index)
        if index in memo:
            return memo[index]
        value = flat[index]
        if isinstance(value, list):
            out: list[Any] = []
            memo[index] = out
            out.extend(walk(v) for v in value)
            return out
        if isinstance(value, dict):
            obj: dict[str, Any] = {}
            memo[index] = obj
            obj.update({k: walk(v) for k, v in value.items()})
            return obj
        memo[index] = value
        return value

    return walk(0)


def trpc_result(body: bytes | str) -> Any:
    """Pull the decoded payload out of a tRPC response body."""
    doc = json.loads(body)
    data = doc["result"]["data"]
    if isinstance(data, str):
        return decode(data)
    # Some routes answer in plain superjson form.
    if isinstance(data, dict) and "json" in data:
        return data["json"]
    return data
