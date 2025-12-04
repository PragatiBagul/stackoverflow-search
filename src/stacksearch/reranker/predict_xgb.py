import xgboost as xgb
import numpy as np
import pandas as pd
from typing import List, Tuple
import json
import os

def load_xgb_model(model_path:str,metadata_path:str=None):
    bst = xgb.Booster()
    bst.load_model(model_path)
    metadata = None
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    return bst, metadata

def rerank_candidates(bst:xgb.Booster,candidates_df:pd.DataFrame,feature_cols:List[str],top_k:int=10) -> pd.DataFrame:
    """
    candidates_df: each row = a (query_id, doc_id, ... features)
    feature_cols: list of features in the same order used for training
    returns: DataFrame sorted by predicted score (descending), with 'score' column
    """
    X = candidates_df[feature_cols].astype(float).values
    dmat = xgb.DMatrix(X)
    scores = bst.predict(dmat)
    candidates_df = candidates_df.copy()
    candidates_df['score'] = scores
    return candidates_df.sort_values('score', ascending=False).head(top_k)