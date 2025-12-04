import faiss
import numpy as np
import os
from typing import Tuple

def build_ivfpq_index(embeddings:np.ndarray,
                      nlist:int = 1024,
                      m: int = 16,
                      nbits: int = 8,
                      use_gpu: bool = False) -> faiss.Index:
    """
    Build an IVF+PQ index.
    - embeddings: np.ndarray (N,D) float32, ideally L2-normalized if using inner product.
    - nlist: number of coarse clusters.
    - m: number of PQ subvectors (must divide D).
    - nbits: bits per subvector (usually 8).

    Returns : trained faiss index (IndexIVFPQ).
    """
    assert embeddings.dtype == np.float32, "embeddings must be float32"
    N, D = embeddings.shape
    if D % m != 0:
        raise ValueError(f"m={m} must divide embedding dim D = {D}")

    # Use inner product for cosine-like similarity when embeddings are normalized
    quantizer = faiss.IndexFlatIP(D)
    index = faiss.IndexIVFPQ(quantizer, D, nlist, m, nbits)

    # Train the coarse quantizer + PQ on a subset (or all) embeddings
    # faiss expects training data as (num_train,D)
    rng = np.random.default_rng(1234)
    num_train = min(max(10000,nlist * 50),N) # Heuristic
    train_idx = rng.choice(N, size=num_train, replace=False)
    train_data = embeddings[train_idx]

    print(f"Training FAISS index with {train_data.shape[0]} vectors")
    index.train(train_data)

    print("Adding embeddings to index...")
    index.add(embeddings)

    return index
def save_faiss_index(index:faiss.Index,path:str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    print(f"Saved FAISS index to {path}")

def load_faiss_index(path:str) -> faiss.Index:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    idx = faiss.read_index(path)
    return idx

def faiss_search(index:faiss.Index, query_embeddings:np.ndarray,top_k:int = 10, nprobe: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Query FAISS index.
    - query_embeddings: shape(Q,D)
    Returns:
        distances (Q,top_k), indices (Q,top_k)
    """
    index.nprobe = nprobe
    distances, indices = index.search(query_embeddings.astype("float32"), top_k)
    return distances, indices

