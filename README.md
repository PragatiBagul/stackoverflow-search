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
