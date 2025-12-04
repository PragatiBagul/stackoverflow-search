# src/stacksearch/features/build_features.py
import numpy as np
import pandas as pd
import pickle
from typing import Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from src.stacksearch.data.preprocess import bm25_tokenize
import os

def load_bm25_artifacts(bm25_dir: str):
    with open(os.path.join(bm25_dir, "bm25_index.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    doc_ids = np.load(os.path.join(bm25_dir, "doc_ids.npy"), allow_pickle=False)
    return bm25, doc_ids

def load_embeddings(emb_path: str, ids_path: str) -> Tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(emb_path)
    doc_ids = np.load(ids_path, allow_pickle=False)
    return embeddings, doc_ids

def ensure_id_to_idx(doc_ids: np.ndarray) -> Dict[int, int]:
    return {int(doc_id): idx for idx, doc_id in enumerate(doc_ids)}

def compute_bm25_score(bm25, query_text: str, doc_idx: int, bm25_tokenize_fn=bm25_tokenize) -> float:
    tokens = bm25_tokenize_fn(query_text)
    scores = bm25.get_scores(tokens)  # aligned with corpus order
    return float(scores[int(doc_idx)])

def compute_token_overlap(query: str, doc_text: str) -> float:
    q_tokens = set(bm25_tokenize(query))
    d_tokens = set(bm25_tokenize(doc_text))
    if len(q_tokens) == 0:
        return 0.0
    return float(len(q_tokens & d_tokens) / len(q_tokens))

def compute_dense_sim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    # both expected to be l2-normalized if possible
    return float(np.dot(query_emb, doc_emb))

def encode_queries_in_batches(queries: pd.Series, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64):
    model = SentenceTransformer(model_name)
    emb_list = []
    for i in range(0, len(queries), batch_size):
        batch = queries.iloc[i:i+batch_size].tolist()
        e = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        # normalize
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        e = e / norms
        emb_list.append(e.astype("float32"))
    return np.vstack(emb_list)

def build_features_from_pairs(pairs_df: pd.DataFrame,
                              questions_meta: pd.DataFrame,
                              bm25,
                              bm25_doc_ids: np.ndarray,
                              embeddings: Optional[np.ndarray],
                              emb_doc_ids: Optional[np.ndarray],
                              st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                              batch_size: int = 64) -> pd.DataFrame:
    """
    inputs:
      - pairs_df: columns ["query_id","query_text","doc_id","label"]
      - questions_meta: DataFrame with columns ["Id","Title","Body", ...]
      - bm25, bm25_doc_ids: BM25 artifacts (bm25 object, aligned doc ids)
      - embeddings, emb_doc_ids: np arrays or None
    returns:
      - features_df: DataFrame with feature columns and labels
    """

    # map doc id -> index in embeddings (if present)
    emb_id_to_idx = None
    if embeddings is not None and emb_doc_ids is not None:
        emb_id_to_idx = ensure_id_to_idx(emb_doc_ids)

    # map bm25 doc ids -> index
    bm25_id_to_idx = ensure_id_to_idx(bm25_doc_ids)

    # meta mapping for token overlap and lengths
    meta_map = questions_meta.set_index("Id")

    # We'll need to batch-encode unique queries to avoid repeated encoding
    unique_queries = pairs_df[["query_id", "query_text"]].drop_duplicates().reset_index(drop=True)
    print(f"Encoding {len(unique_queries)} unique queries in batches (batch_size={batch_size})...")
    query_embeddings = encode_queries_in_batches(unique_queries["query_text"], model_name=st_model_name, batch_size=batch_size)

    query_id_to_emb = {int(qid): query_embeddings[i] for i, qid in enumerate(unique_queries["query_id"])}

    # Build features row-by-row (vectorization is possible but this is clearer)
    rows = []
    for _, r in pairs_df.iterrows():
        qid = int(r["query_id"])
        qtext = r["query_text"]
        did = int(r["doc_id"])
        label = int(r["label"])

        # BM25 score (use bm25 corpus index)
        bm25_idx = bm25_id_to_idx.get(did, None)
        if bm25_idx is None:
            bm25_score = 0.0
        else:
            bm25_score = compute_bm25_score(bm25, qtext, bm25_idx)

        # Token overlap and lengths
        doc_text = ""
        if did in meta_map.index:
            meta_row = meta_map.loc[did]
            doc_text = (meta_row.get("Title", "") or "") + " " + (meta_row.get("Body", "") or "")
        token_overlap = compute_token_overlap(qtext, doc_text)
        query_len = len(bm25_tokenize(qtext))
        doc_len = len(bm25_tokenize(doc_text))

        # Dense similarity: prefer precomputed embeddings if available
        dense_sim = 0.0
        if emb_id_to_idx is not None and did in emb_id_to_idx:
            q_emb = query_id_to_emb[qid]
            d_emb = embeddings[emb_id_to_idx[did]]
            dense_sim = compute_dense_sim(q_emb, d_emb)
        else:
            # If no precomputed doc embedding, fallback to encoding doc on the fly (expensive)
            dense_sim = 0.0

        rows.append({
            "query_id": qid,
            "doc_id": did,
            "query_text": qtext,
            "label": label,
            # features
            "bm25_score": bm25_score,
            "dense_sim": dense_sim,
            "token_overlap": token_overlap,
            "query_len": query_len,
            "doc_len": doc_len
        })

    features_df = pd.DataFrame(rows)
    return features_df
