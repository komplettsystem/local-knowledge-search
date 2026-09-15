#!/usr/bin/env python3
"""Build/refresh the local hybrid search index from ~/.knowledge-base-sync/mirror.

Run this after every mirror sync (wired into run_sync.sh). Cheap to re-run:
only chunks whose content hash changed get re-embedded.
"""
import sys
import time

import numpy as np
from fastembed import TextEmbedding

from lib import INDEX_DIR, load_corpus, save_json

EMBED_CACHE_PATH = INDEX_DIR / "embed_cache.npz"


def main():
    t0 = time.time()
    records = load_corpus()
    if not records:
        print("No markdown files found under the mirror - aborting, leaving old index in place.")
        sys.exit(1)

    cache: dict[str, np.ndarray] = {}
    if EMBED_CACHE_PATH.exists():
        with np.load(EMBED_CACHE_PATH) as data:
            cache = {k: data[k] for k in data.files}

    to_embed = [r for r in records if r["hash"] not in cache]
    if to_embed:
        model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        texts = [r["text"] for r in to_embed]
        vectors = list(model.embed(texts))
        for r, v in zip(to_embed, vectors):
            cache[r["hash"]] = np.asarray(v, dtype=np.float32)

    # Prune cache entries no longer referenced by any current chunk.
    live_hashes = {r["hash"] for r in records}
    cache = {h: v for h, v in cache.items() if h in live_hashes}
    np.savez(EMBED_CACHE_PATH, **cache)

    embeddings = np.stack([cache[r["hash"]] for r in records])
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    save_json(records, "records.json")

    elapsed = time.time() - t0
    print(f"Indexed {len(records)} chunks ({len(to_embed)} newly embedded) in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
