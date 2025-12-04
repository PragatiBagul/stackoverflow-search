# scripts/06_train_reranker.py
import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from src.stacksearch.reranker.train_xgb import train_xgb_ranker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to features_train.parquet")
    parser.add_argument("--output_model", default="artifacts/local/reranker/xgb_reranker.json", help="Where to save model")
    parser.add_argument("--output_meta", default="artifacts/local/reranker/metadata.json", help="Where to save metadata")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of queries for validation")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_boost_round", type=int, default=500)
    parser.add_argument("--early_stopping_rounds", type=int, default=30)
    parser.add_argument("--eval_ndcg_k", type=int, default=10)
    args = parser.parse_args()

    print("📥 Loading features...")
    df = pd.read_parquet(args.features)

    # Validate required columns
    required = {"query_id","doc_id","label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Features file must contain {required}")

    # Feature columns — automatically pick numeric columns excluding id/label/text
    exclude = {"query_id","doc_id","label","query_text"}
    feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    print("Using feature columns:", feature_cols)

    # Split by queries to avoid leakage
    unique_q = df["query_id"].unique()
    train_q, val_q = train_test_split(unique_q, test_size=args.test_size, random_state=args.random_seed)

    train_df = df[df["query_id"].isin(train_q)].reset_index(drop=True)
    val_df = df[df["query_id"].isin(val_q)].reset_index(drop=True)

    print(f"Train queries: {len(train_q)}; Val queries: {len(val_q)}")
    print(f"Train rows: {len(train_df)}; Val rows: {len(val_df)}")

    # Train
    bst, metadata = train_xgb_ranker(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        params=None,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        eval_ndcg_k=args.eval_ndcg_k,
        model_out_path=args.output_model,
        metadata_out_path=args.output_meta
    )

    print("Done. Model saved at:", args.output_model)
    print("Metadata saved at:", args.output_meta)

if __name__ == "__main__":
    main()
