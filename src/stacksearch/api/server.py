# src/stacksearch/api/server.py
import os
import traceback
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd

from src.stacksearch.api.utils import (
    load_questions_meta, load_bm25_index, bm25_search,
    load_embeddings_and_index, query_dense,
    compute_features_for_candidates, merge_candidates, load_st_model
)

# Pydantic request + response models
class SearchRequest(BaseModel):
    query: str
    k: int = 10
    n_bm25: int = 50
    n_dense: int = 50
    rerank: bool = True

class SearchResultItem(BaseModel):
    doc_id: int
    title: str
    snippet: str
    score: float
    source: str  # "bm25" | "dense" | "reranked"

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]

# Application & global objects (loaded on startup)
app = FastAPI(title="StackOverflow Hybrid Search API")

ARTIFACTS = {
    "bm25_dir": os.environ.get("BM25_DIR", "artifacts/local/bm25"),
    "faiss_dir": os.environ.get("FAISS_DIR", "artifacts/local/faiss"),
    "meta_path": os.environ.get("QUESTIONS_META", "data/local/questions_meta.parquet"),
    "reranker_model": os.environ.get("RERANKER_MODEL", "artifacts/local/reranker/xgb_reranker.json"),
    "reranker_meta": os.environ.get("RERANKER_META", "artifacts/local/reranker/metadata.json")
}

# Lazy-loaded resources
_resources = {
    "bm25": None,
    "bm25_doc_ids": None,
    "meta": None,
    "embeddings": None,
    "emb_ids": None,
    "faiss_index": None,
    "st_model": None,
    "reranker": None,
    "reranker_meta": None
}

@app.on_event("startup")
def load_resources():
    # BM25
    try:
        bm25, doc_ids, _ = load_bm25_index(ARTIFACTS["bm25_dir"])
        _resources["bm25"] = bm25
        _resources["bm25_doc_ids"] = doc_ids
        print("BM25 loaded.")
    except Exception as e:
        print("BM25 load error:", e)
        _resources["bm25"] = None

    # Meta table
    try:
        _resources["meta"] = load_questions_meta(ARTIFACTS["meta_path"])
        print("Questions meta loaded.")
    except Exception as e:
        print("Questions meta load failed:", e)
        _resources["meta"] = None

    # FAISS & embeddings (optional)
    try:
        emb, emb_ids, idx = load_embeddings_and_index(ARTIFACTS["faiss_dir"])
        _resources["embeddings"] = emb
        _resources["emb_ids"] = emb_ids
        _resources["faiss_index"] = idx
        print("FAISS index loaded.")
    except Exception as e:
        print("FAISS load failed (ok for local).", e)
        _resources["embeddings"] = None
        _resources["emb_ids"] = None
        _resources["faiss_index"] = None

    # SentenceTransformer model (lazy load when needed)
    _resources["st_model"] = None

    # Reranker model (optional)
    try:
        import xgboost as xgb
        if os.path.exists(ARTIFACTS["reranker_model"]):
            bst = xgb.Booster()
            bst.load_model(ARTIFACTS["reranker_model"])
            _resources["reranker"] = bst
            if os.path.exists(ARTIFACTS["reranker_meta"]):
                with open(ARTIFACTS["reranker_meta"], "r") as f:
                    _resources["reranker_meta"] = json.load(f)
            print("Reranker loaded.")
        else:
            print("Reranker model not found; will skip reranking.")
    except Exception as e:
        print("Reranker load failed:", e)
        _resources["reranker"] = None

@app.get("/health")
def health():
    return {"ok": True, "bm25": _resources["bm25"] is not None, "faiss": _resources["faiss_index"] is not None, "reranker": _resources["reranker"] is not None}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query or len(req.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Empty query")

    try:
        # 1) get BM25 candidates
        bm25_obj = _resources["bm25"]
        bm25_doc_ids = _resources["bm25_doc_ids"]
        if bm25_obj is None:
            raise HTTPException(status_code=500, detail="BM25 not loaded")

        bm25_ids, bm25_scores = bm25_search(bm25_obj, bm25_doc_ids, req.query, k=req.n_bm25, bm25_tokenize_fn=None)

        # 2) get Dense candidates (if available)
        dense_ids = []
        if _resources["faiss_index"] is not None and _resources["st_model"] is None:
            # lazy load st model
            _resources["st_model"] = load_st_model()
        if _resources["faiss_index"] is not None and _resources["st_model"] is not None:
            dists, inds = query_dense(req.query, _resources["faiss_index"], _resources["st_model"], top_k=req.n_dense, nprobe=16)
            # map FAISS indices to doc_ids
            emb_ids = _resources["emb_ids"]
            dense_ids = [int(emb_ids[int(i)]) for i in inds if int(i) != -1]

        # 3) merge candidates
        merged = merge_candidates(list(map(int, bm25_ids)), dense_ids, top_k=max(req.k, 200))

        # 4) compute features for merged candidates
        features_df = compute_features_for_candidates(
            query_text=req.query,
            candidate_doc_ids=merged,
            bm25_obj=bm25_obj,
            bm25_doc_ids=bm25_doc_ids,
            meta_index=_resources["meta"] if _resources["meta"] is not None else pd.DataFrame(),
            embeddings=_resources["embeddings"],
            emb_doc_ids=_resources["emb_ids"],
            st_model=_resources["st_model"]
        )

        # attach source tag (simple heuristic: if in bm25 first 10 -> bm25, if in dense top10 -> dense)
        source_map = {}
        for i, did in enumerate(bm25_ids):
            source_map[int(did)] = "bm25"
        for i, did in enumerate(dense_ids):
            if int(did) not in source_map:
                source_map[int(did)] = "dense"

        # 5) rerank with XGBoost if available & requested
        reranked_df = None
        if req.rerank and _resources["reranker"] is not None and _resources["reranker_meta"] is not None:
            try:
                from src.stacksearch.reranker.predict_xgb import rerank_candidates, load_xgb_model
                bst, _ = load_xgb_model(ARTIFACTS["reranker_model"], ARTIFACTS["reranker_meta"])
                feature_cols = _resources["reranker_meta"]["feature_cols"]
                # Ensure columns exist in features_df; if not, fill 0
                for c in feature_cols:
                    if c not in features_df.columns:
                        features_df[c] = 0.0
                # Prepare candidates df for reranker (need query_id, doc_id, feature cols)
                cand_df = features_df.copy().rename(columns={"query_text":"query_text"})
                cand_df_for_rank = cand_df[["doc_id"] + feature_cols].copy()
                cand_df_for_rank["query_id"] = 0  # single-query batch
                # re-order columns to pass into reranker: include feature cols
                cand_for_pred = cand_df[feature_cols + ["doc_id"]]
                # call reranker
                ranked = rerank_candidates(bst, cand_df, feature_cols, top_k=req.k)
                ranked = ranked.reset_index(drop=True)
                ranked["source"] = ranked["doc_id"].apply(lambda x: source_map.get(int(x), "bm25"))
                result_df = ranked
            except Exception as e:
                print("Reranker failed at query time:", e)
                result_df = features_df.sort_values("bm25_score", ascending=False).head(req.k)
                result_df["source"] = result_df["doc_id"].apply(lambda x: source_map.get(int(x), "bm25"))
        else:
            # No reranker: sort by combined simple score (bm25_score + dense_sim * alpha)
            alpha = 0.7
            features_df["combined_score"] = features_df["bm25_score"] * (1.0 - alpha) + features_df["dense_sim"] * alpha
            result_df = features_df.sort_values("combined_score", ascending=False).head(req.k)
            result_df["source"] = result_df["doc_id"].apply(lambda x: source_map.get(int(x), "bm25"))

        # 6) Format response
        meta = _resources["meta"]
        items = []
        for _, row in result_df.iterrows():
            did = int(row["doc_id"])
            title = ""
            snippet = ""
            if meta is not None and did in meta.index:
                mrow = meta.loc[did]
                title = mrow.get("Title", "") or ""
                snippet = make_snippet_from_body(mrow.get("Body", "") or "")
            items.append({
                "doc_id": did,
                "title": title,
                "snippet": snippet,
                "score": float(row.get("score", row.get("combined_score", row.get("bm25_score", 0.0)))),
                "source": row.get("source", "bm25")
            })

        return {"query": req.query, "results": items}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
