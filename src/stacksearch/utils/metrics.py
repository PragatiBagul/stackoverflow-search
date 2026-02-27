"""
IR metrics used in evaluation of retrieval

Includes:
 - mean_reciprocal_rank
 - recall@k
 - NDCG@K
"""

import math
from typing import List,Set

def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    MRR = 1 / rank of first relevant document in retrieved_ids
    :param retrieved_ids:
    :param relevant_ids:
    :return:
    """
    for i , doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i+1)
    return 0.0

def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Recall@K = (# of relevant documents in retrieved_ids) / (# relevant in truth)
    :param retrieved_ids:
    :param relevant_ids:
    :return:
    """
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids)
    return len(retrieved_set.intersection(relevant_ids)) / len(retrieved_ids)

def dcg_at_k(relevances: List[int])->float:
    """
    Discounted Cumulative Gain over a list of relevance labels
    """
    score = 0.0
    for idx, rel in enumerate(relevances):
        score += (2 ** (rel - 1))/math.log2(idx + 2)
    return score

def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str])->float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K) for binary relevance
    :param retrieved_ids:
    :param relevant_ids:
    :return:
    """
    # 1 if doc is relevant else 0
    rels = [1 if doc in relevant_ids else 0 for doc in retrieved_ids]
    actual_dcg = dcg_at_k(rels)

    # ideal DCG = sort rels decreasing
    ideal = sorted(rels,reverse=True)
    ideal_dcg = dcg_at_k(ideal)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def precision_at_k(
    retrieved_ids: Iterable[str],
    relevant_ids: Set[str],
    k: int | None = None,
) -> float:
    """
    Compute Precision@k.

    Parameters
    ----------
    retrieved_ids : Iterable[str]
        Ranked list of retrieved document IDs (best first).
    relevant_ids : Set[str]
        Set of relevant document IDs.
    k : int, optional
        Cutoff rank. If None, uses full retrieved list.

    Returns
    -------
    float
        Precision@k score in [0, 1].
    """
    if not relevant_ids:
        return 0.0

    retrieved_ids = list(retrieved_ids)

    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    if not retrieved_ids:
        return 0.0

    num_relevant = sum(
        1 for doc_id in retrieved_ids if doc_id in relevant_ids
    )

    return num_relevant / len(retrieved_ids)