"""Unit test: ingestion chunking honors CHUNK_SIZE / CHUNK_OVERLAP."""
import importlib.util
from pathlib import Path


def _load_ingest():
    spec = importlib.util.spec_from_file_location(
        "ingest_under_test",
        Path(__file__).resolve().parents[1] / "ingest.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chunk_size_and_overlap(stub_external_modules):
    ingest = _load_ingest()
    text = "a" * 1200
    chunks = ingest.chunk_text(text, source="t.txt")

    assert chunks, "should produce at least one chunk"
    # All but last chunk should be exactly CHUNK_SIZE chars
    for c in chunks[:-1]:
        assert len(c["text"]) == ingest.CHUNK_SIZE

    # Stride between chunk starts is CHUNK_SIZE - CHUNK_OVERLAP
    stride = ingest.CHUNK_SIZE - ingest.CHUNK_OVERLAP
    # 1200 chars, stride 450, first chunk at 0, second 450, third 900 -> 3 chunks
    expected_count = (len(text) + stride - 1) // stride
    assert len(chunks) == expected_count

    # Metadata + ids
    assert all(c["source"] == "t.txt" for c in chunks)
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_chunk_skips_empty_tail(stub_external_modules):
    ingest = _load_ingest()
    chunks = ingest.chunk_text("hello world", source="x.txt")
    # Single chunk, smaller than CHUNK_SIZE
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world"
