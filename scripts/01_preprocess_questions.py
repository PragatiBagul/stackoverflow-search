import argparse
import pandas as pd
import os

from src.stacksearch.data.preprocess import preprocess_questions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Questions.csv")
    parser.add_argument("--output_dir", default="data/local/", help="Where to write processed data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("📥 Loading input CSV...")
    df = pd.read_csv(args.input, encoding="utf-8")

    print("✨ Preprocessing questions...")
    df_clean = preprocess_questions(df)

    # Save full cleaned dataset
    clean_path = os.path.join(args.output_dir, "questions_clean.parquet")
    df_clean.to_parquet(clean_path, index=False)
    print(f"✅ Saved cleaned questions → {clean_path}")

    # Save metadata for API/snippets
    meta_path = os.path.join(args.output_dir, "questions_meta.parquet")
    df[["Id", "Title", "Body"]].to_parquet(meta_path, index=False)
    print(f"📄 Saved metadata → {meta_path}")

if __name__ == "__main__":
    main()