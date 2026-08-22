"""Byte-faithful container fixtures, built rather than checked in.

Generating them keeps the test suite self-contained and, more usefully, makes the
container layout explicit: the reason these tests exist is that real files put
ComfyUI metadata in five different places, and one of them (mp4 with `moov`
after `mdat`) cost this project a wrong published finding before it was noticed.
"""

from __future__ import annotations

import json
import struct
import zlib

SAVE_GRAPH = {
    "nodes": [
        {
            "id": 1,
            "type": "CheckpointLoaderSimple",
            "pos": [10, 10],
            "mode": 0,
            "widgets_values": ["sd_xl_base_1.0.safetensors"],
            "properties": {"cnr_id": "comfy-core", "ver": "0.3.45"},
            "inputs": [],
            "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1]}],
        },
        {
            "id": 2,
            "type": "KSampler",
            "pos": [300, 10],
            "mode": 0,
            "widgets_values": [42, "randomize", 20, 8.0, "euler", "normal", 1.0],
            "properties": {"cnr_id": "comfy-core", "ver": "0.3.45"},
            "inputs": [{"name": "model", "type": "MODEL", "link": 1}],
            "outputs": [],
        },
        {
            "id": 3,
            "type": "Power Lora Loader (rgthree)",
            "pos": [600, 10],
            "mode": 0,
            "widgets_values": [],
            "properties": {"cnr_id": "rgthree-comfy", "ver": "1.0.0"},
            "inputs": [],
            "outputs": [],
        },
    ],
    "links": [[1, 1, 0, 2, 0, "MODEL"]],
    "groups": [],
    "extra": {"frontendVersion": "1.43.18"},
    "version": 0.4,
    "last_node_id": 3,
    "last_link_id": 1,
}

API_GRAPH = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
    },
    "2": {
        "class_type": "KSampler",
        "inputs": {"seed": 42, "steps": 20, "model": ["1", 0]},
    },
}

WORKFLOW_JSON = json.dumps(SAVE_GRAPH)
PROMPT_JSON = json.dumps(API_GRAPH)


# --- PNG --------------------------------------------------------------------


def _png_chunk(ctype: bytes, body: bytes) -> bytes:
    return (
        struct.pack(">I", len(body))
        + ctype
        + body
        + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF)
    )


def _text_chunk(kind: str, keyword: str, value: str) -> bytes:
    kw = keyword.encode("latin1")
    if kind == "tEXt":
        return _png_chunk(b"tEXt", kw + b"\x00" + value.encode())
    if kind == "zTXt":
        return _png_chunk(b"zTXt", kw + b"\x00\x00" + zlib.compress(value.encode()))
    if kind == "iTXt":
        # compression flag, method, language tag, translated keyword, text
        return _png_chunk(b"iTXt", kw + b"\x00\x00\x00" + b"\x00" + b"\x00" + value.encode())
    raise ValueError(kind)


def png(
    *,
    chunk_kind: str = "tEXt",
    workflow: str | None = WORKFLOW_JSON,
    prompt: str | None = PROMPT_JSON,
    trailing: bool = False,
    padding: int = 0,
) -> bytes:
    """A PNG carrying ComfyUI text chunks. `trailing` puts them after IDAT."""
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
    text = b""
    if prompt is not None:
        text += _text_chunk(chunk_kind, "prompt", prompt)
    if workflow is not None:
        text += _text_chunk(chunk_kind, "workflow", workflow)
    filler = (
        _png_chunk(b"iCCP", b"pad\x00\x00" + zlib.compress(b"\x00" * padding)) if padding else b""
    )
    idat = _png_chunk(b"IDAT", zlib.compress(b"\x00" * 64))
    iend = _png_chunk(b"IEND", b"")
    head = b"\x89PNG\r\n\x1a\x0a" + ihdr
    return head + filler + idat + text + iend if trailing else head + text + filler + idat + iend


# --- WebP -------------------------------------------------------------------


def _riff_chunk(fourcc: bytes, body: bytes) -> bytes:
    return fourcc + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) % 2 else b"")


def _exif_ifd0(tags: dict[int, str]) -> bytes:
    """Little-endian TIFF with ASCII IFD0 entries, values spilled after the IFD."""
    entries = b""
    values = b""
    ifd_offset = 8
    value_base = ifd_offset + 2 + 12 * len(tags) + 4
    for tag, text in sorted(tags.items()):
        raw = text.encode() + b"\x00"
        entries += struct.pack("<HHII", tag, 2, len(raw), value_base + len(values))
        values += raw
    return (
        b"II"
        + struct.pack("<H", 42)
        + struct.pack("<I", ifd_offset)
        + struct.pack("<H", len(tags))
        + entries
        + struct.pack("<I", 0)
        + values
    )


def webp(*, workflow: str | None = WORKFLOW_JSON, prompt: str | None = PROMPT_JSON) -> bytes:
    """Extended WebP whose EXIF chunk sits after the image data, as the spec has it."""
    tags: dict[int, str] = {}
    if workflow is not None:
        tags[0x010F] = f"workflow:{workflow}"
    if prompt is not None:
        tags[0x0110] = f"prompt:{prompt}"
    flags = 0b0000_1000 if tags else 0
    vp8x = _riff_chunk(b"VP8X", bytes([flags, 0, 0, 0]) + b"\x07\x00\x00\x07\x00\x00")
    vp8 = _riff_chunk(b"VP8 ", b"\x00" * 64)
    exif = _riff_chunk(b"EXIF", _exif_ifd0(tags)) if tags else b""
    body = b"WEBP" + vp8x + vp8 + exif
    return b"RIFF" + struct.pack("<I", len(body)) + body


# --- ISOBMFF ----------------------------------------------------------------


def _atom(fourcc: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body) + 8) + fourcc + body


def _moov(payloads: dict[str, str], *, itunes: bool) -> bytes:
    if itunes:
        blob = json.dumps(payloads).encode()
        data = _atom(b"data", struct.pack(">II", 1, 0) + blob)
        ilst = _atom(b"ilst", _atom(b"\xa9cmt", data))
    else:
        entries = b""
        for index, payload in enumerate(payloads.values(), start=1):
            data = _atom(b"data", struct.pack(">II", 1, 0) + payload.encode())
            entries += _atom(struct.pack(">I", index), data)
        ilst = _atom(b"ilst", entries)
    hdlr = _atom(b"hdlr", b"\x00" * 8 + b"mdirappl" + b"\x00" * 9)
    meta = _atom(b"meta", b"\x00\x00\x00\x00" + hdlr + ilst)
    return _atom(b"moov", _atom(b"udta", meta))


def mp4(
    *,
    faststart: bool = False,
    itunes: bool = True,
    workflow: str | None = WORKFLOW_JSON,
    prompt: str | None = PROMPT_JSON,
    mdat_size: int = 200_000,
) -> bytes:
    """An mp4 with ComfyUI metadata. `faststart=False` puts `moov` after `mdat`,
    which is what ffmpeg does by default and why video needs a tail fetch."""
    payloads: dict[str, str] = {}
    if workflow is not None:
        payloads["workflow"] = workflow
    if prompt is not None:
        payloads["prompt"] = prompt
    ftyp = _atom(b"ftyp", b"isomiso2avc1mp41")
    mdat = _atom(b"mdat", b"\x00" * mdat_size)
    moov = _moov(payloads, itunes=itunes) if payloads else b""
    return ftyp + moov + mdat if faststart else ftyp + mdat + moov
