# scripts/05_build_features.py
import argparse
import os
import pandas as pd
import numpy as np
import json
from src.stacksearch.features.build_features import (
    load_bm25_artifacts, load_embeddings, build_features_from_pairs
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="Path to pairs_train.parquet")
    parser.add_argument("--questions_meta", required=True, help="Path to questions_meta.parquet")
    parser.add_argument("--bm25_dir", default="artifacts/local/bm25", help="BM25 artifacts dir")
    parser.add_argument("--embeddings", default="artifacts/local/faiss/doc_embeddings.npy", help="Embeddings npy path (optional)")
    parser.add_argument("--emb_ids", default="artifacts/local/faiss/doc_ids.npy", help="Embedding doc ids npy path (optional)")
    parser.add_argument("--output", default="data/local/features_train.parquet", help="Output features parquet path")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    print("📥 Loading pairs...")
    pairs = pd.read_parquet(args.pairs)

    print("📥 Loading questions metadata...")
    questions_meta = pd.read_parquet(args.questions_meta)

    print("📥 Loading BM25 artifacts...")
    bm25, bm25_doc_ids = load_bm25_artifacts(args.bm25_dir)

    embeddings = None
    emb_ids = None
    if os.path.exists(args.embeddings) and os.path.exists(args.emb_ids):
        print("📥 Loading embeddings and doc ids...")
        embeddings = np.load(args.embeddings)
        emb_ids = np.load(args.emb_ids, allow_pickle=False)
    else:
        print("⚠️ Embeddings not found — dense features will be zero. Proceeding...")

    print("🔧 Building features for pairs (this can take some time)...")
    features = build_features_from_pairs(
        pairs_df=pairs,
        questions_meta=questions_meta,
        bm25=bm25,
        bm25_doc_ids=bm25_doc_ids,
        embeddings=embeddings,
        emb_doc_ids=emb_ids,
        batch_size=args.batch_size
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    features.to_parquet(args.output, index=False)
    print(f"✅ Features saved to {args.output}")
    print(features.head())

if __name__ == "__main__":
    main()
