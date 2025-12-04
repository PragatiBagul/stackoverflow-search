import pickle
from pathlib import Path
from typing import List,Tuple,Optional
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.stacksearch.data.preprocess import bm25_tokenize

def build_bm25_index(questions_df:pd.DataFrame) -> Tuple[BM25Okapi,np.ndarray,List[List[str]]]:
    """
    Build a BM25Okapi index from questions dataframe
    Returns:
    - bm25: BM25Okapi index
    - doc_ids: numpy array of document ids aligned with bm25 corpus order
    - corpus_tokens: list of token lists (same order)
    """
    # Expect questions_df to have columns : Id, text_cleaned (Title_Body cleaned)
    if 'Id' not in questions_df.columns or 'text_cleaned' not in questions_df.columns:
        raise ValueError('questions_df must have Id,text_cleaned column')

    # Align doc ids and texts in order
    doc_ids = questions_df['Id'].to_numpy()
    corpus = questions_df['text_cleaned'].astype(str).tolist()

    # Tokenize once
    corpus_tokens = [bm25_tokenize(t) for t in corpus]

    # Build BM25
    bm25 = BM25Okapi(corpus_tokens)
    return bm25, doc_ids, corpus_tokens

def save_bm25_artifacts(bm25,doc_ids:np.ndarray,corpus_tokens:List[List[str]],output_dir:str,prefix:Optional[str] = 'bm25') -> None:
    """
    Persist bm25 object and doc ids (and optionally corpus tokens) to disk
    Files :
    - {output_dir}/{prefix}_index.pkl
    - {output_dir}/{prefix}_doc_ids.pkl
    - {output_dir}/{prefix}_corpus_tokens.pkl
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    index_path = outdir / f'{prefix}_index.pkl'
    doc_ids_path = outdir / f'{prefix}_doc_ids.npy'
    corpus_tokens_path = outdir / f'{prefix}_corpus_tokens.pkl'

    # Save pickle BM25
    with index_path.open('wb') as f:
        pickle.dump(bm25, f)

    # Save doc ids
    np.save(str(doc_ids_path), doc_ids)

    # Save corpus tokens (optional)
    with corpus_tokens_path.open('wb') as f:
        pickle.dump(corpus_tokens, f)

def load_bm25_artifacts(input_dir:str,prefix:Optional[str] = 'bm25'):
        """
        Load bm25, doc_ids, corpus_tokens from disk
        Returns bm25, doc_ids, corpus_tokens
        """
        in_dir = Path(input_dir)
        index_path = in_dir / f'{prefix}_index.pkl'
        doc_ids_path = in_dir / f'{prefix}_doc_ids.npy'
        corpus_tokens_path = in_dir / f'{prefix}_corpus_tokens.pkl'

        # Save pickle BM25
        with index_path.open('rb') as f:
            bm25 = pickle.load(f)
        doc_ids = np.load(str(doc_ids_path),allow_pickle=False)

        corpus_tokens = None
        if corpus_tokens_path.exists():
            import pickle as _p
            with corpus_tokens_path.open('rb') as f:
                corpus_tokens = _p.load(f)
        return bm25, doc_ids, corpus_tokens
def bm25_search(bm25,doc_ids:np.ndarray,query_text:str,k:int=10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run BM25 search and return top-k doc ids and scores

    Returns:
        - top_doc_ids: numpy array of top doc ids (lenght <= k)
        - top_scores : numpy array of corresponding scores
    """
    tokens = bm25_tokenize(query_text)
    scores = bm25.get_scores(tokens)

    # argsort descending
    top_idx = scores.argsort()[::-1][:k]
    top_doc_ids = doc_ids[top_idx]
    top_scores = scores[top_idx]
    return top_doc_ids, top_scores