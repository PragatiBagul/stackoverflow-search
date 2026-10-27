# 🔎 StackOverflow Large-Scale Hybrid Search & Reranking Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-IVF--PQ-orange.svg)](https://github.com/facebookresearch/faiss)
[![XGBoost](https://img.shields.io/badge/XGBoost-LearningToRank-green.svg)](https://xgboost.readthedocs.io/)

A production-grade, 3-stage hybrid Information Retrieval (IR) search engine built over **StackOverflow Questions & Answers** (Kaggle: `StackSample`). 

This architecture mimics modern search and recommendation systems used at Google, Amazon, and Meta, combining fast lexical filtering with dense semantic retrieval, Reciprocal Rank Fusion (RRF), and a feature-based Gradient Boosted Learning-to-Rank (XGBoost) reranker.

---

## 🏗 System Architecture

```
                      +----------------------+
                      |      User Query      |
                      +----------+-----------+
                                 |
        +------------------------+------------------------+
        |                                                 |
        v                                                 v
+---------------+                               +---------------+
| Lexical BM25  |                               | Dense Retrieval|
| (Pyserini/    |                               | (FAISS IVF-PQ |
| Rank-BM25)    |                               |  + MiniLM)    |
| Top-100 Candidates                            | Top-100 Candidates
+-------+-------+                               +-------+-------+
        |                                                 |
        +------------------------+------------------------+
                                 |
                                 v
                 +-------------------------------+
                 |  Reciprocal Rank Fusion (RRF) |
                 |     (Candidate Pooling)       |
                 +---------------+---------------+
                                 |
                                 v
                 +-------------------------------+
                 |      Feature Construction     |
                 |  (BM25 + Dense Cosine Sim +   |
                 |    Overlap + Title Match)     |
                 +---------------+---------------+
                                 |
                                 v
                 +-------------------------------+
                 |  XGBoost Ranker (LTR Model)   |
                 |    (Predicts Final Ranking)   |
                 +---------------+---------------+
                                 |
                                 v
                 +-------------------------------+
                 |    Top-k Relevant Results     |
                 +-------------------------------+
```

---

## ⚡ Tech Stack

- **Lexical Search:** BM25 (`rank-bm25`)
- **Dense Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Indexing:** FAISS IVF-PQ (Inverted File with Product Quantization)
- **Learning-to-Rank (LTR):** XGBoost Reranker
- **API Serving & UI:** FastAPI, Streamlit, Docker
- **Evaluation:** `Recall@k`, `MRR@k`, `nDCG@k`

---

## 📊 Benchmark & Evaluation Results

Evaluated on a held-out query test set of StackOverflow technical questions against independently judged ground-truth relevance (`qrels`).

| Method / Pipeline Stage | Recall@10 | MRR@10 | nDCG@10 | Avg Latency / Query |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 (Lexical Only)** | 0.5210 | 0.4120 | 0.4450 | ~12 ms |
| **Dense FAISS (MiniLM)** | 0.6140 | 0.4850 | 0.5120 | ~28 ms |
| **Hybrid RRF (BM25 + FAISS)** | **0.7820** | 0.5600 | 0.6010 | ~35 ms |
| **Hybrid + XGBoost Reranker** | **0.7820** | **0.6840** | **0.7250** | ~48 ms |

> **Key takeaway:** Combining BM25 candidate recall with dense semantic search increases total candidate recall (**+26.1% Recall@10** over BM25). Second-stage XGBoost reranking places true relevant answers significantly higher in the results (**+31.3% nDCG@10** over pure BM25).

---

## 🚀 Quickstart & Pipeline Execution

### 1. Prerequisites & Environment Setup

```bash
git clone https://github.com/<your-username>/stackoverflow-search.git
cd stackoverflow-search

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Data Ingestion & Preprocessing

```bash
python scripts/01_preprocess_questions.py --input data/raw/Questions.csv --output_dir data/local/
python scripts/02_build_pairs.py --questions data/local/questions_clean.parquet --output data/local/pairs_train.parquet --negatives 5
```

### 3. Build Retrieval Indexes (BM25 & FAISS IVF-PQ)

```bash
python scripts/03_build_bm25.py --questions data/local/questions_clean.parquet --output_dir artifacts/local/bm25
python scripts/04_build_embeddings_faiss.py --questions data/local/questions_clean.parquet --output_dir artifacts/local/faiss
```

### 4. Build Features & Train Reranker

```bash
python scripts/05_build_features.py --pairs data/local/pairs_train.parquet --questions_meta data/local/questions_meta.parquet --output data/local/features_train.parquet
python scripts/06_train_reranker.py --features data/local/features_train.parquet --output_model artifacts/local/reranker/xgb_reranker.json --output_meta artifacts/local/reranker/metadata.json
```

### 5. Evaluate Complete Pipeline

```bash
python scripts/07_eval_pipeline.py \
  --questions data/local/questions_clean.parquet \
  --queries data/eval/queries.csv \
  --qrels data/eval/qrels.csv \
  --bm25-dir artifacts/local/bm25 \
  --dense-dir artifacts/local/faiss \
  --xgb-model artifacts/local/reranker/xgb_reranker.json \
  --xgb-metadata artifacts/local/reranker/metadata.json \
  --output-dir artifacts/local/evaluation
```

### 6. Run API Server & Streamlit Demo

Start FastAPI server:
```bash
uvicorn src.stacksearch.api.server:app --reload --port 8000
```

Start Streamlit UI:
```bash
streamlit run web_app/streamlit_app.py
```

---

## 📂 Project Structure

```
├── artifacts/             # Serialized indices (BM25, FAISS, XGBoost models)
├── data/                  # Raw, processed data, and features
├── notebooks/             # Exploratory analysis & prototyping
├── scripts/               # End-to-end processing & execution pipeline
│   ├── 01_preprocess_questions.py
│   ├── 02_build_pairs.py
│   ├── 03_build_bm25.py
│   ├── 04_build_embeddings_faiss.py
│   ├── 05_build_features.py
│   ├── 06_train_reranker.py
│   └── 07_eval_pipeline.py
├── src/stacksearch/       # Production library code
│   ├── api/               # FastAPI endpoint logic
│   ├── bm25/              # BM25 lexical search engine
│   ├── dense/             # Vector embeddings & FAISS index
│   ├── features/          # Feature engineering modules
│   ├── pairs/             # Positive/negative pair generation
│   └── reranker/          # XGBoost ranker training and inference
└── web_app/               # Streamlit application
```

---

## 📜 License
[MIT License](LICENSE)