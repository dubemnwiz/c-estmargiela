#!/usr/bin/env python
import argparse
from collections import Counter

import faiss
import numpy as np
import pandas as pd


def eval_retrieval(
    embeddings: np.ndarray,
    garment_ids,
    ks=(1, 3, 5, 10),
):
    """
    embeddings: (N, d) L2-normalized vectors
    garment_ids: list/array of length N
    """
    N, d = embeddings.shape
    garment_ids = np.array(garment_ids)

    # Build index in-memory for eval
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))

    # Count garments
    counts = Counter(garment_ids.tolist())
    valid_mask = np.array([counts[g] >= 2 for g in garment_ids])
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        raise ValueError("No garments have at least 2 images; cannot compute retrieval metrics.")

    print(f"[INFO] Evaluating on {len(valid_indices)} images (garments with >=2 images).")

    ks = sorted(ks)
    max_k = ks[-1] + 1  # +1 to accommodate self-removal
    recall_at_k = {k: 0 for k in ks}
    mrr_sum = 0.0
    n_queries = 0

    for idx in valid_indices:
        q = embeddings[idx : idx + 1]
        true_garment = garment_ids[idx]

        # search including self
        D, I = index.search(q, max_k + 1)
        neighbors = I[0].tolist()

        # remove self index
        neighbors = [n for n in neighbors if n != idx]

        # rank of first correct neighbor
        rank = None
        for r, n in enumerate(neighbors[:max_k], start=1):
            if garment_ids[n] == true_garment:
                rank = r
                break

        # Recall@K
        for k in ks:
            topk = neighbors[:k]
            if any(garment_ids[n] == true_garment for n in topk):
                recall_at_k[k] += 1

        # MRR
        if rank is not None:
            mrr_sum += 1.0 / rank

        n_queries += 1

    recall_at_k = {k: recall_at_k[k] / n_queries for k in ks}
    mrr = mrr_sum / n_queries
    return recall_at_k, mrr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--item-images-csv",
        default="data/catalog/margiela/item_images.csv",
        help="Path to item_images.csv",
    )
    parser.add_argument(
        "--embeddings-path",
        default="data/derivs/embeddings/margiela_items_siglip_embeddings.npy",
        help="Path to saved numpy embeddings from SigLIP.",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="K values for Recall@K.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.item_images_csv)
    garment_ids = df["garment_id"].tolist()

    embeddings = np.load(args.embeddings_path)
    print(f"[INFO] Loaded embeddings {embeddings.shape}")

    recall_at_k, mrr = eval_retrieval(
        embeddings=embeddings,
        garment_ids=garment_ids,
        ks=args.ks,
    )

    print("\n=== SigLIP Item Retrieval Metrics ===")
    for k in sorted(recall_at_k.keys()):
        print(f"Recall@{k}: {recall_at_k[k]:.3f}")
    print(f"MRR: {mrr:.3f}")


if __name__ == "__main__":
    main()
