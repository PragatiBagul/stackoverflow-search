# web_app/streamlit_app.py
import os
import time
import json
from typing import List, Dict, Any, Optional

import streamlit as st
import requests
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Try to import local search utils (for fallback). If your repo structure is different,
# Streamlit must be started from the project root so Python can import src.
LOCAL_SEARCH_AVAILABLE = True
try:
    from src.stacksearch.bm25.build_bm25 import load_bm25_index, bm25_search
    from src.stacksearch.api.utils import make_snippet_from_body, load_questions_meta
except Exception:
    LOCAL_SEARCH_AVAILABLE = False

# ---------------------
# Config
# ---------------------
API_URL = os.environ.get("SO_SEARCH_API", "http://localhost:8000")  # expects /search
DEFAULT_BM25_DIR = os.environ.get("BM25_DIR", "artifacts/local/bm25")
DEFAULT_META = os.environ.get("QUESTIONS_META", "data/local/questions_meta.parquet")

st.set_page_config(page_title="StackOverflow Hybrid Search (Demo)", layout="wide")

# ---------------------
# Helpers
# ---------------------
def call_search_api(query: str, k: int, n_bm25: int, n_dense: int, rerank: bool, api_url: str = API_URL):
    payload = {
        "query": query,
        "k": k,
        "n_bm25": n_bm25,
        "n_dense": n_dense,
        "rerank": rerank
    }
    try:
        t0 = time.time()
        r = requests.post(f"{api_url}/search", json=payload, timeout=15)
        latency = time.time() - t0
        r.raise_for_status()
        return r.json(), latency, None
    except Exception as e:
        return None, None, e

def local_bm25_search(query: str, k: int, bm25_dir: str = DEFAULT_BM25_DIR, meta_path: str = DEFAULT_META):
    """
    BM25-only fallback: loads bm25 artifacts + questions_meta,
    returns results list similar to API response.
    """
    bm25, doc_ids, _ = load_bm25_index(bm25_dir)
    top_ids, scores = bm25_search(bm25, doc_ids, query, k=k, bm25_tokenize_fn=None)
    meta = load_questions_meta(meta_path)
    items = []
    for did, sc in zip(top_ids.tolist(), scores.tolist()):
        did = int(did)
        title = ""
        snippet = ""
        if did in meta.index:
            row = meta.loc[did]
            title = row.get("Title", "") or ""
            snippet = make_snippet_from_body(row.get("Body", "") or "")
        items.append({"doc_id": did, "title": title, "snippet": snippet, "score": float(sc), "source": "bm25"})
    return {"query": query, "results": items}

# ---------------------
# UI layout
# ---------------------
st.title("🔎 StackOverflow Hybrid Search — Demo UI")

col1, col2 = st.columns([3,1])
with col1:
    query = st.text_input("Search query (title-like text)", value="", placeholder="e.g. python json parse, java NullPointerException", key="query_input")
    query_submit = st.button("Search")

with col2:
    k = st.slider("k (results)", 1, 50, 10)
    n_bm25 = st.slider("BM25 candidates", 1, 200, 50)
    n_dense = st.slider("Dense candidates", 0, 200, 50)
    rerank = st.checkbox("Use reranker (requires API & model)", value=True)
    st.markdown(" ")
    st.caption("API endpoint: " + API_URL)

st.write("---")

# Status area
status_placeholder = st.empty()
results_placeholder = st.empty()

# Preload local resources lazily (only on fallback)
_local_loaded = False

if query_submit and query.strip():
    status_placeholder.info("Running search…")
    # 1) Attempt API call
    api_resp, latency, api_err = call_search_api(query, k, n_bm25, n_dense, rerank)
    if api_resp is not None:
        # Display results
        status_placeholder.success(f"API results returned in {latency*1000:.0f} ms")
        df = pd.DataFrame(api_resp.get("results", []))
        with results_placeholder.container():
            st.subheader(f"Results (API) — {len(df)}")
            for i, row in df.iterrows():
                st.markdown(f"**{i+1}. {row['title'] if row.get('title') else row['doc_id']}**  ")
                st.write(row.get("snippet", ""))
                st.markdown(f"Score: `{row.get('score', 0.0):.4f}` • Source: `{row.get('source','')}`")
                st.write("")
    else:
        # API failed — show error and try local BM25 fallback if available
        status_placeholder.warning(f"API call failed: {api_err}. Falling back to local BM25 (if available).")
        if LOCAL_SEARCH_AVAILABLE:
            try:
                t0 = time.time()
                resp = local_bm25_search(query, k, bm25_dir=DEFAULT_BM25_DIR, meta_path=DEFAULT_META)
                latency_local = time.time() - t0
                status_placeholder.success(f"Local BM25 returned in {latency_local*1000:.0f} ms")
                df = pd.DataFrame(resp.get("results", []))
                with results_placeholder.container():
                    st.subheader(f"Results (BM25 fallback) — {len(df)}")
                    for i, row in df.iterrows():
                        st.markdown(f"**{i+1}. {row['title'] if row.get('title') else row['doc_id']}**  ")
                        st.write(row.get("snippet", ""))
                        st.markdown(f"Score: `{row.get('score', 0.0):.4f}` • Source: `{row.get('source','')}`")
                        st.write("")
            except Exception as e:
                status_placeholder.error(f"Local fallback failed: {e}")
                st.exception(e)
        else:
            status_placeholder.error("Local search utilities are not importable. Make sure Streamlit is run from project root and src is on PYTHONPATH.")
            st.markdown("**To run fallback,** start Streamlit from the project root where `src/stacksearch` exists.")

st.markdown("---")
st.caption("This demo calls the local API at `SO_SEARCH_API` (env) if running. Otherwise it uses a BM25 fallback if available. Use it to sanity-check results quickly.")
