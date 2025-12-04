import numpy as np
from sentence_transformers import SentenceTransformer
from src.stacksearch.dense.faiss_index import load_faiss_index, faiss_search
import faiss
def load_embeddings_and_index(artifacts_dir:str):
    emb = np.load(f"{artifacts_dir}/doc_embeddings.npy")
    ids = np.load(f"{artifacts_dir}/doc_ids.npy")
    idx = load_faiss_index(f"{artifacts_dir}/faiss_index.ivfpq")
    return emb, ids, idx

def query_dense(query_text:str,index,st_model:SentenceTransformer,top_k:int=10,nprobe:int=16):
    q_emb = st_model.encode([query_text],normalize_embeddings=True,convert_to_numpy=True).astype("float32")
    index.nprobe = nprobe
    distances, indices = index.search(q_emb,top_k=top_k)
    return distances[0], indices[0]


