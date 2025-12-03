import argparse
import pandas as pd
import os
from src.stacksearch.pairs.build_pairs import build_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', required=True,help='Path to questions_clean.parquet')
    parser.add_argument('--output', required=True,help='Path to save pairs_train.parquet')
    parser.add_argument('--negatives',type=int,default=5,help='Number of negative samples per query')
    args = parser.parse_args()

    print('Loading preprocessed questions...')
    questions = pd.read_parquet(args.questions)

    print('Building pairs...')
    pairs = build_pairs(questions, n_negatives=args.negatives)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pairs.to_parquet(args.output,index=False)
    print(f"Saved Pairs -> {args.output}")
    print(pairs.head())

if __name__ == '__main__':
    main()