import os
import pickle
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Tuple, List, Dict, Any, Optional

# Local imports
from src.stacksearch.bm25.build_bm25 import load_bm25_artifacts, bm25_search
from src.stacksearch.dense.search import load_embeddings_and_index, query_dense
from src.stacksearch.data.preprocess import bm25_tokenize

# Small snippet extractor
def make_snippet_from_body(body:str,max_chars:int=250)->str:
    if not isinstance(body,str) or len(body.strip()) == 0:
        return ""
    text = " ".join(body.split())
    if len(text) <= max_chars:
        return text
    # try to cut at sentence boundary
    cut = text[:max_chars]
    last_period = cut.rfind(". ")
    if last_period != -1 and last_period > max_chars // 2:
        return cut[:last_period+1]
    return cut + "..."

# Load metadata table (Id, Title, Body)
def load_questions_meta(path:str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    meta = pd.read_parquet(path)
    return meta.set_index("id")

# Simple feature computation for candidates
def compute_features_for_candidates(
        query_text:str,
        candidate_doc_ids : List[int],
        bm25_obj,
        bm25_doc_ids:np.ndarray,
        meta_index:pd.DataFrame,
        embeddings:Optional[np.ndarray]=None,
        emb_doc_ids:Optional[np.ndarray]=None,
        st_model:Optional[SentenceTransformer]=None
)->pd.DataFrame:
    """
    Returns DataFrame with columns:
    query_id (same as query text hash), doc_id, bm25_score, dense_sim, taken_overlap, query_len, doc_len
    """
    rows = []
    # build mapping index -> position for bm25 & embeddings
    bm25_id_to_idx = {int(i): idx for idx, i in enumerate(bm25_doc_ids)}
    emb_id_to_idx = None
    if embeddings is not None and emb_doc_ids is not None:
        emb_id_to_idx = {int(i): idx for idx, i in enumerate(emb_doc_ids)}

    # compute query embedding once if needed
    q_emb = None
    if st_model is not None:
        q_emb = st_model.encode([query_text],convert_to_numpy=True, normalize_embeddings=True)[0].astype('float32')

    q_tokens = set(bm25_tokenize(query_text))
    q_len = len(q_tokens)

    # BM25 full score vector (for fast lookup) - returns scores aligned bm25_doc_ids order
    try:
        bm25_scores_all = bm25_obj.get_scores(bm25_tokenize(query_text))
    except Exception as e:
        # fallback compute per candidate
        bm25_scores_all = None

    for did in candidate_doc_ids:
        did_int = int(did)
        # BM25 score
        bm25_score = 0.0
        if bm25_scores_all is not None and did_int in bm25_id_to_idx:
            bm25_score = float(bm25_scores_all[bm25_id_to_idx[did_int]])
        else:
            # Try get via bm25_search single doc
            try:
                idx = bm25_id_to_idx.get(did_int,None)
                if idx is not None:
                    bm25_score = float(bm25_obj.get_scores(bm25_tokenize(query_text))[idx])
            except Exception as e:
                bm25_score = 0.0

        # token overlap & lengths (Title + Body)
        doc_text = ""
        if did_int in meta_index.index:
            row = meta_index.loc[did_int]
            doc_text = (row.get('Title','') or '') + " " + (row.get('Body','') or '')
        d_tokens = set(bm25_tokenize(doc_text))
        token_overlap = float(len(q_tokens & d_tokens) / len(q_tokens)) if q_len > 0 else 0.0
        doc_len = len(d_tokens)

        # dense sim if available
        dense_sim = 0.0
        if q_emb is not None and emb_id_to_idx is not None and did_int in emb_id_to_idx:
            d_emb = embeddings[emb_id_to_idx[did_int]]
            dense_sim = float(np.dot(q_emb, d_emb))
        rows.append({
            "query_text": query_text,
            "doc_id":did_int,
            "bm25_score":bm25_score,
            "dense_sim":dense_sim,
            "token_overlap":token_overlap,
            "query_len":q_len,
            "doc_len":doc_len,
        })
    return pd.DataFrame(rows)

# Merge candidates (bm25 + dense) preserving order/scoring optionally
def merge_candidates(
    bm25_ids: List[int],
    dense_ids: List[int],
    top_k: int = 100
) -> List[int]:
    # simple union preserving BM25 order first, then add dense in their order
    seen = set()
    merged = []
    for did in bm25_ids:
        if int(did) not in seen:
            merged.append(int(did))
            seen.add(int(did))
            if len(merged) >= top_k:
                return merged
    for did in dense_ids:
        if int(did) not in seen:
            merged.append(int(did))
            seen.add(int(did))
            if len(merged) >= top_k:
                return merged
    return merged[:top_k]

# Helper: load sentence-transformer only if needed
def load_st_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)
