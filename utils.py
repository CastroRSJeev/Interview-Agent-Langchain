import os
import re
import json
from datetime import datetime, timedelta
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

os.environ.setdefault("USER_AGENT", "InterviewAgent2/1.0")

CACHE_DIR = "corpus_cache"
CACHE_TTL_DAYS = 14

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")


def normalize_tech_name(tech_name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', tech_name.lower())


def get_cache_paths(tech_key: str):
    entry_dir = os.path.join(CACHE_DIR, tech_key)
    return {
        "dir": entry_dir,
        "faiss_index": os.path.join(entry_dir, "faiss_index"),
        "corpus_txt": os.path.join(entry_dir, "corpus.txt"),
        "meta_json": os.path.join(entry_dir, "meta.json"),
    }


def load_cached_vectorstore(tech_name: str):
    """Load a cached FAISS vectorstore for this tech if present and fresh. Returns None if missing/stale."""
    tech_key = normalize_tech_name(tech_name)
    paths = get_cache_paths(tech_key)

    if not os.path.exists(paths["meta_json"]):
        return None

    try:
        with open(paths["meta_json"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        cached_at = datetime.fromisoformat(meta["cached_at"])
        if datetime.now() - cached_at > timedelta(days=CACHE_TTL_DAYS):
            print(f"  Cache for '{tech_name}' is stale (>{CACHE_TTL_DAYS}d old) — refreshing.")
            return None
        vectorstore = FAISS.load_local(
            paths["faiss_index"], embeddings, allow_dangerous_deserialization=True
        )
        print(f"  Cache hit for '{tech_name}' (cached {meta['cached_at']}) — skipping Tavily/scrape.")
        return vectorstore
    except Exception as e:
        print(f"  Cache read failed for '{tech_name}': {e} — will refetch.")
        return None


def save_cache(tech_name: str, vectorstore, corpus_text: str, source_urls: list):
    tech_key = normalize_tech_name(tech_name)
    paths = get_cache_paths(tech_key)
    os.makedirs(paths["dir"], exist_ok=True)

    vectorstore.save_local(paths["faiss_index"])

    with open(paths["corpus_txt"], "w", encoding="utf-8") as f:
        f.write(corpus_text)

    meta = {
        "tech_name": tech_name,
        "cached_at": datetime.now().isoformat(),
        "source_urls": source_urls,
    }
    with open(paths["meta_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)