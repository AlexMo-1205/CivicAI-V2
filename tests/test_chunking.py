"""Unit tests: chunk_text honors chunk_size / chunk_overlap and the small-doc guard."""
from civicai.config import SETTINGS
from civicai.rag.ingest import chunk_text


def test_chunk_size_and_overlap_when_doc_exceeds_split_threshold():
    # Force the sliding-window path: text must be strictly larger than the guard.
    text = "a" * (SETTINGS.min_chunk_split_chars + 1000)
    chunks = chunk_text(text, source="big.txt")

    assert len(chunks) > 1, "doc above the split threshold must be chunked"
    # Every non-tail chunk should be exactly chunk_size chars
    for c in chunks[:-1]:
        assert len(c["text"]) == SETTINGS.chunk_size

    stride = SETTINGS.chunk_size - SETTINGS.chunk_overlap
    expected_count = (len(text) + stride - 1) // stride
    assert len(chunks) == expected_count

    assert all(c["source"] == "big.txt" for c in chunks)
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_small_doc_stays_as_single_chunk():
    """Short procedural docs must NOT be split — steps and penalties stay together."""
    text = "a" * (SETTINGS.min_chunk_split_chars - 10)
    chunks = chunk_text(text, source="small.txt")

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == 0
    assert chunks[0]["source"] == "small.txt"
    assert chunks[0]["text"] == text


def test_chunk_short_input():
    chunks = chunk_text("hello world", source="x.txt")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world"
    assert chunks[0]["chunk_id"] == 0


def test_empty_input_returns_no_chunks():
    assert chunk_text("", source="empty.txt") == []
    assert chunk_text("   \n\t  ", source="ws.txt") == []
