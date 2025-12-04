import xgboost as xgb
import numpy as np
import pandas as pd
import json
import os
from typing import Tuple, List, Dict
import math

# Helper metrics for ranking
def get_rank_in_list(pos_doc,retrieved_ids):
    try:
        return retrieved_ids.index(pos_doc) + 1
    except ValueError:
        return None

def ndcg_at_k_binary_for_query(pos_doc,retrieved_ids,k=10):
    dcg = 0.0
    for i,did in enumerate(retrieved_ids[:k]):
        rel = 1 if did == pos_doc else 0
        if rel:
            dcg += 1.0/math.log2(i+2)
    idcg = 1.0
    return dcg / idcg

def compute_ranking_metrics_for_predictions(df_pred:pd.DataFrame, k=10) -> Dict[str,float]:
    """
    df_pred must have columns : query_id, doc_id, label, score
    For each query, sort by score desc to get retrieved_ids.
    Returns Recall@k, MRR@k, nDCG@k (binary relevance)
    """
    grouped = df_pred.groupby('query_id')
    recalls = []
    rr_list = []
    ndcgs = []

    for qid, group in  grouped:
        group_sorted = group.sort_values("score", ascending=False)
        retrieved = group_sorted['doc_id'].tolist()
        pos_docs = group_sorted[group_sorted['label'] == 1]['doc_id'].tolist()
        # If multiple positives, treat first positive doc as target for MRR, for recall treat any
        # We'll compute binary metrics with single-target assumption (first positive)
        target = pos_docs[0] if len(pos_docs) > 0 else None

        if target is None:
            # no positive in group: skip metrics
            continue
        # Hit / Recall@K
        hit = 1 if target in retrieved[:k] else 0
        recalls.append(hit)

        # MRR
        rank = get_rank_in_list(target, retrieved[:k])
        rr = 1.0 / rank if rank is not None and rank > 0 else 0.0
        rr_list.append(rr)

        # nDCG@k
        ndcg = ndcg_at_k_binary_for_query(target,retrieved,k)
        ndcgs.append(ndcg)
    if len(recalls) == 0:
        return {"recall@k":0.0,"mrr@k":0.0,"ndcg@k":0.0}
    return {
        "recall@k":float(np.mean(recalls)),
        "mrr@k":float(np.mean(rr_list)),
        "ndcg@k":float(np.mean(ndcgs))
    }

# Data preparation helpers
def prepare_dmatrix_from_features(df:pd.DataFrame,feature_cols:List[str]) -> Tuple[xgb.DMatrix,List[int]]:
    """
    Input df must be grouped by query (any order), with columns:
    - query_id, doc_id, label, [feature_cols...]

    Returns:
        - dmatrix : XGBoost DMatrix with labels
        - group : list of ints (number of candidates per query) in the same order as X matrix
    """
    df_sorted = df.sort_values(['query_id']).reset_index(drop=True)
    groups = df_sorted.groupby('query_id').size().tolist()

    X = df_sorted[feature_cols].astype(np.float32).values
    y = df_sorted['label'].astype(np.float32).values

    dmat = xgb.DMatrix(X, label=y)
    dmat.set_groups(groups)
    return dmat, groups

# Train function
def train_xgb_ranker(train_df:pd.DataFrame,
                    val_df:pd.DataFrame,
                    feature_cols:List[str],
                    params:Dict = None,
                    num_boost_round:int = 1000,
                    early_stopping_rounds:int = 50,
                    eval_ndcg_k: int = 10,
                    model_out_path:str = "artifacts/local/reranker/xgb_reranker.json",
                    metadata_out_path:str = "artifacts/local/reranker/metadata.json"):
    """
    Args:
        train_df, val_df : DataFrames with query_id, doc_id, and label + feature_cols
        feature_cols : list of features to use in XGBoost
        params : xgboost parameters dict (if None, sane defaults used)
    Returns:
        trained xgb.Booster object
    """
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)

    if params is None:
        params = {
            "objective":"rank:ndcg",
            "eval_metric":"ndcg",
            "eta":0.05,
            "max_depth":6,
            "subsample":0.8,
            "colsample_bytree":0.8,
            "verbosity":1,
        }
    print("Preparing DMatrix for training")
    dtrain,train_groups = prepare_dmatrix_from_features(train_df,feature_cols)
    dval, val_groups = prepare_dmatrix_from_features(val_df, feature_cols)

    evals_result = {}
    watchlist = [(dtrain,'train'),(dval,'validation')]

    print("Starting training with params: ",params)
    bst = xgb.train(params=params,dtrain=dtrain,num_boost_round=num_boost_round,evals=watchlist,early_stopping_rounds=early_stopping_rounds,verbose_eval=10,evals_result=evals_result)

    # Save model
    print(f"Saving model to {model_out_path}")
    bst.save_model(model_out_path)

    # Evaluate on val_df using the model predictions
    print("Computing val predictions for ranking metrics...")
    # Build feature matrix in the same order as prepare_dmatrix
    val_sorted = val_df.sort_values(['query_id']).reset_index(drop=True)
    X_val = val_sorted[feature_cols].astype(np.float32).values
    dmatrix_val = xgb.DMatrix(X_val)
    preds = bst.predict(dmatrix_val)

    val_sorted['score'] = preds
    metrics = compute_ranking_metrics_for_predictions(val_sorted[["query_id","doc_id","label","score"]],k=eval_ndcg_k)
    print("Validation metrics: ",metrics)

    # Save metadata
    metadata = {
        "feature_cols":feature_cols,
        "params":params,
        "num_boost_round":len(bst.get_dump()),
        "val_metrics":metrics,
    }
    with open(metadata_out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return bst, metadata