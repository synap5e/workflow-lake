"""Parser tests, one per way the wild actually stores a workflow."""

from __future__ import annotations

import json

import pytest
from fixtures import API_GRAPH, PROMPT_JSON, SAVE_GRAPH, WORKFLOW_JSON, mp4, png, webp

from lake.wfmeta import classify_workflow, parse


@pytest.mark.parametrize("kind", ["tEXt", "zTXt", "iTXt"])
def test_png_text_chunk_types(kind: str) -> None:
    """ComfyUI writes tEXt, but re-encoders emit zTXt and iTXt and a tEXt-only
    parser misses those silently."""
    result = parse(png(chunk_kind=kind))
    assert result.fmt == "png"
    assert json.loads(result.workflow) == SAVE_GRAPH
    assert json.loads(result.prompt) == API_GRAPH
    assert {r.kind for r in result.records} == {kind}


def test_png_stops_at_pixel_data() -> None:
    result = parse(png(), scan_past_image_data=False)
    assert result.workflow is not None
    # needed_bytes is the whole point of the Range strategy: it is where the
    # fetcher could have stopped.
    assert result.needed_bytes is not None
    assert result.needed_bytes < len(png())


def test_png_without_metadata_is_a_cheap_negative() -> None:
    blob = png(workflow=None, prompt=None)
    result = parse(blob, scan_past_image_data=False)
    assert not result.has_any
    # The verdict must land at the first pixel chunk, not at end of file: that
    # is what makes negatives cost ~2 KB instead of a full download.
    assert result.image_data_at is not None
    assert result.image_data_at < 200


def test_png_truncated_head_reports_truncation_so_the_fetcher_escalates() -> None:
    """A short window can land mid-chunk: the small `prompt` fits, the larger
    `workflow` does not. The parser must say so rather than report absence, or
    the fetcher stops one window early and silently loses the graph."""
    blob = png(workflow="x" * 4000, padding=400_000)
    result = parse(blob[:1024], scan_past_image_data=False)
    assert result.truncated
    assert result.prompt is not None
    assert result.workflow is None
    assert "truncated" in result.reason


def test_png_trailing_chunks_are_found_only_when_scanning_past() -> None:
    blob = png(trailing=True)
    assert parse(blob, scan_past_image_data=False).workflow is None
    full = parse(blob, scan_past_image_data=True)
    assert json.loads(full.workflow) == SAVE_GRAPH
    assert all(r.after_image_data for r in full.records)


def test_webp_exif() -> None:
    result = parse(webp())
    assert result.fmt == "webp"
    assert json.loads(result.workflow) == SAVE_GRAPH
    assert json.loads(result.prompt) == API_GRAPH


def test_webp_without_exif_flag_is_negative() -> None:
    result = parse(webp(workflow=None, prompt=None))
    assert not result.has_any
    assert "EXIF" in result.reason


def test_mp4_faststart_is_found_in_the_head() -> None:
    result = parse(mp4(faststart=True))
    assert result.fmt == "isobmff"
    assert json.loads(result.workflow) == SAVE_GRAPH
    assert not result.needs_tail


def test_mp4_moov_after_mdat_asks_for_the_tail() -> None:
    """The bug that produced a wrong published finding: a head-only read of this
    layout finds nothing and looks exactly like 'the source strips metadata'."""
    blob = mp4(faststart=False)
    head = parse(blob[: 128 * 1024])
    assert not head.has_any
    assert head.needs_tail, "head must signal that the tail is where the answer is"

    tail = parse(blob[-128 * 1024 :])
    assert tail.fmt == "isobmff-tail"
    assert json.loads(tail.workflow) == SAVE_GRAPH


def test_mp4_mdta_layout_classifies_bare_payloads_by_shape() -> None:
    """mdta-style files store one bare graph per box with the names in a sibling
    `keys` atom, so the graph's own shape has to say which is which."""
    result = parse(mp4(faststart=True, itunes=False))
    assert json.loads(result.workflow) == SAVE_GRAPH
    assert json.loads(result.prompt) == API_GRAPH


def test_unknown_container_is_a_clean_negative() -> None:
    result = parse(b"GIF89a" + b"\x00" * 100)
    assert not result.has_any
    assert "unsupported" in result.reason


def test_hostile_chunk_length_does_not_allocate() -> None:
    blob = bytearray(png())
    blob[8:12] = (2**31).to_bytes(4, "big")  # absurd IHDR length
    result = parse(bytes(blob))
    assert not result.has_any


@pytest.mark.parametrize(
    "graph,expected",
    [(SAVE_GRAPH, "save"), (API_GRAPH, "api"), ({"a": 1}, "unknown"), ([], "unknown")],
)
def test_classify_workflow(graph, expected) -> None:
    assert classify_workflow(graph) == expected


def test_json_is_required_not_just_a_keyword_match() -> None:
    result = parse(png(workflow="not json at all", prompt=PROMPT_JSON))
    assert result.workflow is None
    assert result.prompt is not None
    assert any(r.keyword == "workflow" and not r.is_json for r in result.records)


def test_payloads_round_trip_unchanged() -> None:
    result = parse(png())
    assert result.workflow == WORKFLOW_JSON
