"""Sanity: config exposes every magic value the rest of the code relies on."""
from pathlib import Path

from civicai.config import SETTINGS, build_system_prompt


def test_settings_constants_are_locked():
    assert SETTINGS.model == "claude-sonnet-4-5"
    assert SETTINGS.embed_model == "BAAI/bge-m3"
    assert SETTINGS.embed_dim == 1024
    assert SETTINGS.chunk_size == 500
    assert SETTINGS.chunk_overlap == 50
    # Collection name encodes model+dim so a model swap won't mix spaces.
    assert SETTINGS.collection_name == "civicai_bge_m3_1024"
    assert SETTINGS.similarity_threshold == 0.5
    assert SETTINGS.default_n_results == 5
    assert SETTINGS.max_tokens == 4096
    assert SETTINGS.tavily_max_results == 5


def test_retrieval_pipeline_settings():
    assert SETTINGS.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert SETTINGS.retrieve_top_k == 40
    assert SETTINGS.rerank_top_n == 8
    # Phase 3 sweep replaces this placeholder.
    assert 0.0 <= SETTINGS.rerank_routing_threshold <= 1.0


def test_settings_paths_anchored_to_project_root():
    assert isinstance(SETTINGS.db_dir, Path)
    assert SETTINGS.db_dir.name == "chroma_db"
    assert SETTINGS.docs_dir.name == "docs"
    assert SETTINGS.static_dir.name == "static"
    assert SETTINGS.static_index.name == "index.html"


def test_system_prompt_keeps_french_routing_rule():
    prompt = build_system_prompt()
    # Routing rule must remain verbatim — this is the contract with Claude.
    assert "search_docs en premier" in prompt
    assert "score < 0.5" in prompt
    assert "web_search" in prompt
