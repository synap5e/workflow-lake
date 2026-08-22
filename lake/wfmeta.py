"""Byte-level extraction of ComfyUI workflow metadata from image containers.

Ported from workflow-badge-extension's TypeScript parsers (src/lib/{png,webp}.ts)
and widened by what the E1/E2 probes actually hit in the wild:

- PNG carries text in three chunk types, not one. ComfyUI's own `SaveImage`
  writes uncompressed `tEXt`, but re-encoders (PIL round-trips, Civitai-side
  tooling, custom save nodes) emit `zTXt` (zlib) and `iTXt` (UTF-8, optionally
  zlib). A tEXt-only parser silently misses those.
- WebP keeps ComfyUI JSON in EXIF IFD0 ASCII tags 0x010F/0x0110 prefixed
  `workflow:` / `prompt:`.
- JPEG: same EXIF layout inside an APP1 segment, plus a COM fallback.
- ISOBMFF (mp4/mov): ComfyUI's video save path goes through ffmpeg, which parks
  the graph in a `©cmt` comment inside `moov/udta/meta/ilst`. Unlike PNG, `moov`
  is routinely written *after* `mdat` — so for video the metadata is at the END
  of the file and a head-only Range fetch finds nothing. This is the real
  "written after the image data" case; PNG never exhibited it in the sample.

Every parse records the absolute byte offset at which each payload *ends*, which
is what the Range-cutoff experiment (E2) is measured from.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any

PNG_SIG = b"\x89PNG\r\n\x1a\n"

# Refuse to allocate on an absurd declared chunk length (hostile/corrupt input).
CHUNK_SANITY_CAP = 512 * 1024 * 1024


@dataclass
class TextRecord:
    """One text payload found in a container."""

    keyword: str
    kind: str  # tEXt | zTXt | iTXt | exif | com
    value_len: int
    start: int  # absolute offset of the chunk header
    end: int  # absolute offset just past the chunk (incl. CRC where applicable)
    is_json: bool
    after_image_data: bool = False


@dataclass
class ParseResult:
    fmt: str | None = None
    workflow: str | None = None
    prompt: str | None = None
    records: list[TextRecord] = field(default_factory=list)
    # True when the byte slice ran out before the parser could conclude.
    truncated: bool = False
    # Offset at which the parser could have stopped and still had everything.
    needed_bytes: int | None = None
    # Offset of the first pixel-data chunk (PNG IDAT / WebP VP8*), if reached.
    image_data_at: int | None = None
    # ISOBMFF only: the container puts its metadata at the end of the file, so
    # this slice cannot answer the question — fetch the tail.
    needs_tail: bool = False
    reason: str = ""

    @property
    def has_any(self) -> bool:
        return self.workflow is not None or self.prompt is not None

    def keywords(self) -> list[str]:
        return [r.keyword for r in self.records]


def _is_json(value: str) -> bool:
    try:
        json.loads(value)
    except Exception:
        return False
    return True


def _take(rec: ParseResult, keyword: str, value: str, kind: str, start: int, end: int) -> None:
    ok = _is_json(value)
    rec.records.append(
        TextRecord(
            keyword=keyword,
            kind=kind,
            value_len=len(value),
            start=start,
            end=end,
            is_json=ok,
            after_image_data=rec.image_data_at is not None,
        )
    )
    if not ok:
        return
    if keyword == "workflow" and rec.workflow is None:
        rec.workflow = value
        rec.needed_bytes = max(rec.needed_bytes or 0, end)
    elif keyword == "prompt" and rec.prompt is None:
        rec.prompt = value
        rec.needed_bytes = max(rec.needed_bytes or 0, end)


def sniff(data: bytes) -> str | None:
    if data.startswith(PNG_SIG):
        return "png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:2] == b"\xff\xd8":
        return "jpeg"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "isobmff"  # avif / heic / mp4
    return None


def parse(data: bytes, *, scan_past_image_data: bool = True) -> ParseResult:
    """Parse a (possibly truncated) prefix of an image file.

    `scan_past_image_data=False` mirrors the extension's cheap mode: stop at the
    first PNG IDAT. True keeps walking so trailing-chunk writers are detected.
    """
    rec = ParseResult(fmt=sniff(data))
    if rec.fmt == "png":
        _parse_png(data, rec, scan_past_image_data)
    elif rec.fmt == "webp":
        _parse_webp(data, rec)
    elif rec.fmt == "jpeg":
        _parse_jpeg(data, rec)
    elif rec.fmt == "isobmff":
        _parse_isobmff(data, rec)
    elif rec.fmt is None and (b"moov" in data[:4096] or b"ilst" in data or b"udta" in data):
        # A tail slice of an mp4 has no signature to sniff — it starts mid-file.
        rec.fmt = "isobmff-tail"
        scan_ilst_data_boxes(data, rec)
        if not rec.has_any:
            rec.reason = "ISOBMFF tail carried no ComfyUI metadata"
    else:
        rec.reason = f"unsupported container: {rec.fmt or 'unknown'}"
    return rec


def _parse_png(data: bytes, rec: ParseResult, scan_past: bool) -> None:
    off = 8
    n = len(data)
    while True:
        if off + 8 > n:
            rec.truncated = True
            rec.reason = "truncated mid-header"
            return
        (length,) = struct.unpack_from(">I", data, off)
        ctype = data[off + 4 : off + 8].decode("latin1")
        if length > CHUNK_SANITY_CAP:
            rec.reason = f"insane {ctype} length {length}"
            return
        end = off + 8 + length + 4  # header + data + CRC
        if ctype in ("IDAT", "fdAT"):
            if rec.image_data_at is None:
                rec.image_data_at = off
            if not scan_past:
                rec.reason = "stopped at IDAT"
                return
        if ctype == "IEND":
            rec.reason = "clean EOF"
            return
        if ctype in ("tEXt", "zTXt", "iTXt"):
            if end > n:
                rec.truncated = True
                rec.reason = f"truncated inside {ctype}"
                return
            body = data[off + 8 : off + 8 + length]
            for kw, val in _png_text_payloads(ctype, body):
                _take(rec, kw, val, ctype, off, end)
        off = end
        if off > n:
            rec.truncated = True
            rec.reason = "truncated mid-chunk"
            return


def _png_text_payloads(ctype: str, body: bytes) -> list[tuple[str, str]]:
    nul = body.find(b"\x00")
    if nul <= 0:
        return []
    keyword = body[:nul].decode("latin1")
    rest = body[nul + 1 :]
    try:
        if ctype == "tEXt":
            return [(keyword, rest.decode("utf-8", "replace"))]
        if ctype == "zTXt":
            # 1 byte compression method, then zlib stream
            return [(keyword, zlib.decompress(rest[1:]).decode("utf-8", "replace"))]
        if ctype == "iTXt":
            # compression flag, method, lang tag \0, translated keyword \0, text
            if len(rest) < 2:
                return []
            flag = rest[0]
            cursor = 2
            for _ in range(2):
                nxt = rest.find(b"\x00", cursor)
                if nxt < 0:
                    return []
                cursor = nxt + 1
            payload = rest[cursor:]
            if flag:
                payload = zlib.decompress(payload)
            return [(keyword, payload.decode("utf-8", "replace"))]
    except Exception:
        return []
    return []


def _parse_webp(data: bytes, rec: ParseResult) -> None:
    off = 12
    n = len(data)
    saw_vp8x = False
    while True:
        if off + 8 > n:
            rec.truncated = True
            rec.reason = "truncated mid-chunk-header"
            return
        fourcc = data[off : off + 4].decode("latin1")
        (length,) = struct.unpack_from("<I", data, off + 4)
        if length > CHUNK_SANITY_CAP:
            rec.reason = f"insane {fourcc} length {length}"
            return
        body_start = off + 8
        end = body_start + length + (length % 2)
        if fourcc == "VP8X":
            saw_vp8x = True
            if body_start < n:
                if not (data[body_start] & 0b0000_1000):
                    rec.reason = "VP8X EXIF flag unset"
                    return
        elif fourcc in ("VP8 ", "VP8L", "ANMF"):
            if rec.image_data_at is None:
                rec.image_data_at = off
            if not saw_vp8x:
                rec.reason = "simple WebP, EXIF impossible"
                return
        elif fourcc == "EXIF":
            if end > n:
                rec.truncated = True
                rec.reason = "truncated inside EXIF"
                return
            _parse_exif(data[body_start : body_start + length], rec, off, end)
        off = end
        if off >= n:
            rec.truncated = off > n
            rec.reason = "end of slice"
            return


def _parse_jpeg(data: bytes, rec: ParseResult) -> None:
    off = 2
    n = len(data)
    while off + 4 <= n:
        if data[off] != 0xFF:
            rec.reason = "desync"
            return
        marker = data[off + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            off += 2
            continue
        if marker == 0xDA:  # start of scan == pixel data
            rec.image_data_at = off
            rec.reason = "stopped at SOS"
            return
        (seglen,) = struct.unpack_from(">H", data, off + 2)
        end = off + 2 + seglen
        if end > n:
            rec.truncated = True
            rec.reason = "truncated mid-segment"
            return
        body = data[off + 4 : end]
        if marker == 0xE1 and body.startswith(b"Exif\x00\x00"):
            _parse_exif(body[6:], rec, off, end)
        elif marker == 0xFE:  # COM
            text = body.decode("utf-8", "replace")
            for kw in ("workflow", "prompt"):
                if text.startswith(kw + ":"):
                    _take(rec, kw, text[len(kw) + 1 :], "com", off, end)
        off = end
    rec.truncated = True
    rec.reason = "ran out of slice"


def _iter_atoms(data: bytes, start: int, end: int):
    """Yield (fourcc, body_start, body_end, atom_start) for one ISOBMFF level."""
    off = start
    while off + 8 <= end:
        (size,) = struct.unpack_from(">I", data, off)
        fourcc = data[off + 4 : off + 8].decode("latin1", "replace")
        if size == 1:  # 64-bit extended size
            if off + 16 > end:
                return
            (size,) = struct.unpack_from(">Q", data, off + 8)
            body_start = off + 16
        elif size == 0:  # extends to end of file
            size = end - off
            body_start = off + 8
        else:
            body_start = off + 8
        if size < 8:
            return
        yield fourcc, body_start, min(off + size, end), off
        off += size


def _parse_isobmff(
    data: bytes, rec: ParseResult, *, _depth: int = 0, _start: int = 0, _end: int | None = None
) -> None:
    """Walk mp4/mov atoms far enough to answer two questions.

    1. Is the metadata here at all, or is `moov` parked after `mdat` (in which
       case only a tail fetch can answer)?
    2. If it is here, what do the `data` boxes hold?

    The payload extraction is a scan for `data` boxes rather than a strict
    descent through moov/udta/meta/ilst, because both iTunes-style (`©cmt`) and
    mdta-style (`keys` + indexed `ilst`) layouts occur in ComfyUI output and both
    put the bytes in a `data` box. The scan also works on a bare tail slice,
    which has no `ftyp` to descend from.
    """
    end = len(data) if _end is None else _end
    saw_mdat = saw_moov = False
    for fourcc, _body_start, _body_end, atom_start in _iter_atoms(data, _start, end):
        if fourcc == "mdat":
            saw_mdat = True
            if rec.image_data_at is None:
                rec.image_data_at = atom_start
        elif fourcc == "moov":
            saw_moov = True
    scan_ilst_data_boxes(data, rec)
    if not rec.has_any:
        rec.needs_tail = saw_mdat and not saw_moov
        rec.reason = (
            "moov after mdat; tail needed"
            if rec.needs_tail
            else "no ComfyUI metadata in ISOBMFF atoms"
        )


def scan_ilst_data_boxes(data: bytes, rec: ParseResult) -> None:
    """Find every `data` box in the slice and try its payload as ComfyUI JSON."""
    pos = 0
    while True:
        idx = data.find(b"data", pos)
        if idx < 0:
            return
        pos = idx + 4
        if idx < 4:
            continue
        (size,) = struct.unpack_from(">I", data, idx - 4)
        if size < 16 or size > CHUNK_SANITY_CAP:
            continue
        box_end = idx - 4 + size
        if box_end > len(data):
            continue
        _take_isobmff_payload(data[idx + 4 + 8 : box_end], rec, idx - 4, box_end)


def _take_isobmff_payload(payload: bytes, rec: ParseResult, start: int, end: int) -> None:
    """Decode one `data` box payload.

    Two layouts occur, and the box gives no hint which is which:

    * **iTunes-style** — a single `©cmt` box holding the wrapper ffmpeg is given,
      `{"prompt": "<json string>", "workflow": "<json string>"}`;
    * **mdta-style** — one box per metadata key, each holding a *bare* graph. The
      key names live in a sibling `keys` atom, but the graph's own shape says
      which it is just as reliably: a canvas graph has `nodes`, a prompt graph is
      a map of `class_type` entries.
    """
    text = payload.decode("utf-8", "replace").strip("\x00").strip()
    if not text.startswith("{"):
        return
    try:
        obj = json.loads(text)
    except Exception:
        return
    if not isinstance(obj, dict):
        return
    wrapped = False
    for keyword in ("workflow", "prompt"):
        value = obj.get(keyword)
        if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            _take(rec, keyword, value, "mp4-ilst", start, end)
            wrapped = True
        elif isinstance(value, (dict, list)):
            _take(rec, keyword, json.dumps(value), "mp4-ilst", start, end)
            wrapped = True
    if wrapped:
        return
    kind = classify_workflow(obj)
    if kind == "save":
        _take(rec, "workflow", text, "mp4-ilst", start, end)
    elif kind == "api":
        _take(rec, "prompt", text, "mp4-ilst", start, end)


def _parse_exif(blob: bytes, rec: ParseResult, chunk_start: int, chunk_end: int) -> None:
    if blob[:4] == b"Exif":
        blob = blob[6:]
    if len(blob) < 8:
        return
    order = blob[:2]
    if order == b"II":
        e = "<"
    elif order == b"MM":
        e = ">"
    else:
        return
    if struct.unpack_from(e + "H", blob, 2)[0] != 42:
        return
    (ifd,) = struct.unpack_from(e + "I", blob, 4)
    if ifd + 2 > len(blob):
        return
    (count,) = struct.unpack_from(e + "H", blob, ifd)
    for i in range(count):
        entry = ifd + 2 + i * 12
        if entry + 12 > len(blob):
            break
        tag, typ, cnt = struct.unpack_from(e + "HHI", blob, entry)
        if typ != 2:  # ASCII
            continue
        if tag not in (0x010F, 0x0110, 0x9286, 0x9C9C, 0x010E):
            continue
        val_off = entry + 8 if cnt <= 4 else struct.unpack_from(e + "I", blob, entry + 8)[0]
        if val_off + cnt > len(blob):
            continue
        raw = blob[val_off : val_off + cnt].rstrip(b"\x00").decode("utf-8", "replace")
        sep = raw.find(":")
        if sep < 0:
            continue
        _take(rec, raw[:sep], raw[sep + 1 :], "exif", chunk_start, chunk_end)


# --- workflow shape helpers -------------------------------------------------


def classify_workflow(obj: Any) -> str:
    """'save' (litegraph canvas graph), 'api' (prompt format), or 'unknown'."""
    if not isinstance(obj, dict):
        return "unknown"
    if "nodes" in obj and isinstance(obj.get("nodes"), list):
        return "save"
    if obj and all(
        isinstance(v, dict) and "class_type" in v for v in obj.values() if isinstance(v, dict)
    ):
        # API format: {"3": {"class_type": ..., "inputs": {...}}, ...}
        if any(isinstance(v, dict) and "class_type" in v for v in obj.values()):
            return "api"
    return "unknown"
