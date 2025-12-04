import argparse
import os
import numpy as np
import pandas as pd
from src.stacksearch.dense.embed import embed_texts
from src.stacksearch.dense.faiss_index import build_ivfpq_index, save_faiss_index
from src.stacksearch.data.preprocess import bm25_tokenize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions",required=True,help="Path to questions_clean parquet")
    parser.add_argument("--output_dir",default="artifacts/local/faiss",help="Where to store embeddings and index")
    parser.add_argument("--model_name",default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--nlist",type=int,default=1024)
    parser.add_argument("--m",type=int,default=16)
    parser.add_argument("--nbits",type=int,default=8)
    parser.add_argument("--nprobe",type=int,default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("📥 Loading questions...")
    questions = pd.read_parquet(args.questions)
    texts = questions['text_cleaned'].fillna("").astype(str).tolist()
    doc_ids = questions['Id'].to_numpy()

    print("🧠 Creating embeddings (this can take a while)...")
    embeddings = embed_texts(texts,model_name=args.model_name, batch_size=args.batch_size,normalize=True)
    emb_path = os.path.join(args.output_dir, "doc_embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"Save embeddings to {emb_path}")

    print("🧭 Building FAISS IVF+PQ index...")
    index = build_ivfpq_index(embeddings,nlist=args.nlist,m=args.m,nbits=args.nbits)
    idx_path = os.path.join(args.output_dir, "faiss_index.ivfpq")
    save_faiss_index(index,idx_path)

    # Save doc_ids mapping
    print(f'Doc ids  : {doc_ids}')
    ids_path = os.path.join(args.output_dir,"doc_ids.npy")
    np.save(ids_path, doc_ids)
    print(f"Saved doc_ids -> {ids_path}")

    # Save nprobe setting for later
    meta = {"nprobe":args.nprobe, "nlist":args.nlist,"m":args.m,"nbits":args.nbits}
    import json
    with open(os.path.join(args.output_dir,"index_meta.json"),"w") as f:
        json.dump(meta,f)
    print("✅ Done.")

if __name__ == "__main__":
    main()