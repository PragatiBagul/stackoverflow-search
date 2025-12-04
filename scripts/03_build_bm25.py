import argparse
import os
from pathlib import Path
import pandas as pd
from src.stacksearch.bm25.build_bm25 import build_bm25_index, save_bm25_artifacts, bm25_search
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', required=True, help='Path to questions_clean.parquet (input)')
    parser.add_argument('--output_dir', default='artifacts/local/bm25', help='Directory to write bm25_index.pkl. doc_ids.npy')
    parser.add_argument("--prefix",default="bm25",help='Filename prefix')
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        raise FileNotFoundError(f"{questions_path} not found")

    print("Loading preprocessed questions...")
    questions = pd.read_parquet(str(questions_path))

    print("Building BM25 index...(this may take a minute for 10K docs")
    bm25, doc_ids, corpus_tokens = build_bm25_index(questions)

    print("Built BM25 over {len(doc_ids)} docs")

    print("Saving artifacts to {args.output_dir}")
    save_bm25_artifacts(bm25, doc_ids, corpus_tokens, args.output_dir,prefix=args.prefix)
    print("Saved BM25 index and doc_ids")

    for i in range(min(3,len(questions))):
        q_title = questions.iloc[i]['Title']
        q_id = questions.iloc[i]['Id']
        top_ids, top_scores = bm25_search(bm25,doc_ids,q_title,k=5)
        print(f"\n Query title (id={q_id}) : {q_title}")
        print("Top results (id,score")
        for tid, s in zip(top_ids,top_scores):
            print(f"{tid},{s}")

if __name__ == "__main__":
    main()
