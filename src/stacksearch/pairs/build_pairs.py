import pandas as pd
import numpy as np
from typing import List, Optional


def build_pairs(questions: pd.DataFrame, n_negatives: int = 5) -> pd.DataFrame:
    """
    Build query–document pairs for training and evaluation.

    Each question becomes:
    - 1 positive pair:  (query = question title, doc_id = question Id, label=1)
    - N negative pairs: doc_id randomly sampled from other questions (label=0)

    Args:
        questions: DataFrame with columns ["Id", "Title", "text_cleaned"]
        n_negatives: number of negatives per query

    Returns:
        DataFrame with columns ["query_id", "query_text", "doc_id", "label"]
    """

    questions = questions.copy()

    # Ensure required columns exist
    required = {"Id", "Title", "text_cleaned"}
    if not required.issubset(set(questions.columns)):
        raise ValueError(f"Questions df must contain {required}")

    all_doc_ids = questions["Id"].tolist()
    id_to_title = dict(zip(questions["Id"], questions["Title"]))

    pairs = []

    for _, row in questions.iterrows():
        q_id = row["Id"]
        q_text = row["Title"]
        pos_doc = q_id

        # Positive pair
        pairs.append({
            "query_id": q_id,
            "query_text": q_text,
            "doc_id": pos_doc,
            "label": 1
        })

        # Negative samples
        neg_candidates = [i for i in all_doc_ids if i != q_id]
        neg_sample = np.random.choice(
            neg_candidates,
            size=min(n_negatives, len(neg_candidates)),
            replace=False
        )

        for neg in neg_sample:
            pairs.append({
                "query_id": q_id,
                "query_text": q_text,
                "doc_id": int(neg),
                "label": 0
            })

    df_pairs = pd.DataFrame(pairs)
    return df_pairs
