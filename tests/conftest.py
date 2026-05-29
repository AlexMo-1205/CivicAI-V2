"""Test config: provide stub API keys so module-level dotenv loads don't bark."""
import os


os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
