"""Unit test: chunk_text honors SETTINGS.chunk_size / chunk_overlap."""
from civicai.config import SETTINGS
from civicai.rag.ingest import chunk_text


def test_chunk_size_and_overlap():
    text = "a" * 1200
    chunks = chunk_text(text, source="t.txt")

    assert chunks, "expected at least one chunk"
    # Every non-tail chunk should be exactly chunk_size chars
    for c in chunks[:-1]:
        assert len(c["text"]) == SETTINGS.chunk_size

    stride = SETTINGS.chunk_size - SETTINGS.chunk_overlap
    expected_count = (len(text) + stride - 1) // stride
    assert len(chunks) == expected_count

    assert all(c["source"] == "t.txt" for c in chunks)
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_chunk_short_input():
    chunks = chunk_text("hello world", source="x.txt")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world"
    assert chunks[0]["chunk_id"] == 0
