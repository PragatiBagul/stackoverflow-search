from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from typing import List, Iterable

def embed_texts(texts: Iterable[str],
                model_name:str = "sentence-transformers/all-MiniLM-L6-v2",
                batch_size:int = 64,
                normalize:bool = True)-> np.ndarray:
    """
    Encode a list/Series of texts to dense embeddings using SentenceTransformers.
    Returns numpy array shape (N,D) dtype=float32

    Normalizes embeddings to unit length when normalize=True (recommended for cosine).
    """
    model = SentenceTransformer(model_name)
    all_embs:List[np.ndarray] = []
    texts = list(texts)

    for i in tqdm(range(0,len(texts),batch_size),desc="Embedding batches"):
        batch = texts[i:min(i+batch_size,len(texts))]
        emb = model.encode(batch,show_progress_bar=True,convert_to_numpy=True)
        if normalize:
            # L2 normalize
            norms = np.linalg.norm(emb,axis=1,keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms
        all_embs.append(emb.astype('float32'))
    return np.vstack(all_embs)


