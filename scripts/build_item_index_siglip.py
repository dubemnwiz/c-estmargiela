#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import faiss # Facebook library for efficient similarity search
import numpy as np # Numerical computing
import open_clip # OpenAI's CLIP model library
import pandas as pd # Data manipulation
import torch # PyTorch deep learning framework
from PIL import Image # Image loading and processing
from tqdm import tqdm # Progress bar display

def load_siglip(
    model_name: str = "ViT-B-16-SigLIP",
    pretrained: str = "webli",
    device: str = "cpu",
):
    """
    Load SigLIP via OpenCLIP's built-in weights.

    model_name: one of the OpenCLIP SigLIP models, e.g.
        - "ViT-B-16-SigLIP"
        - "ViT-B-16-SigLIP-256"
        - "ViT-SO400M-14-SigLIP"

    pretrained: usually "webli" for SigLIP variants.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained=pretrained,
    )
    model.eval()
    model.to(device)
    return model, preprocess


def embed_images(
    df: pd.DataFrame, 
    device: str = "cpu", 
    batch_size: int = 16,
):
    model, preprocess = load_siglip(device=device)
    image_paths = df["image_path"].tolist()

    all_feats = []
    bad_indices = []

    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Embedding images"):
            batch_paths = image_paths[start : start + batch_size]
            images = []
            valid_indices = []

            for i, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(preprocess(img))
                    valid_indices.append(start + i)
                except Exception as e:
                    print(f"[WARN] Failed to load {p}: {e}")
                    bad_indices.append(start + i)

            if not images:
                continue

            images_tensor = torch.stack(images).to(device)
            feats = model.encode_image(images_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
            all_feats.append(feats.cpu().numpy())

    if not all_feats:
        raise RuntimeError("No embeddings were computed; check your image paths.")

    if bad_indices:
        print(f"[INFO] Skipped {len(bad_indices)} images due to load errors.")

    embeddings = np.concatenate(all_feats, axis=0)
    return embeddings


def build_faiss_index(embeddings: np.ndarray, metric: str = "ip"):
    d = embeddings.shape[1]
    if metric == "ip":
        index = faiss.IndexFlatIP(d)
    elif metric == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    index.add(embeddings.astype(np.float32))
    return index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--item-images-csv",
        default="data/catalog/margiela/item_images.csv",
        help="Path to item_images.csv",
    )
    parser.add_argument(
        "--embeddings-out",
        default="data/derivs/embeddings/margiela_items_siglip_embeddings.npy",
        help="Where to save numpy embeddings.",
    )
    parser.add_argument(
        "--index-out",
        default="data/index/margiela_items_siglip.faiss",
        help="Where to save FAISS index.",
    )
    parser.add_argument(
        "--meta-out",
        default="data/index/margiela_items_siglip_meta.json",
        help="Where to save metadata sidecar.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda | cpu",
    )
    parser.add_argument(
        "--metric",
        default="ip",
        choices=["ip", "l2"],
        help="Similarity metric for FAISS index.",
    )
    parser.add_argument(
        "--model-name",
        default="ViT-B-16-SigLIP",
        help="Model name for open_clip (one of the OpenCLIP SigLIP models).",
    )
    parser.add_argument(
        "--pretrained",
        default="webli",
        help="Pretrained tag for open_clip (often None when using hf-hub).",
    )
    args = parser.parse_args()

    # Ensure dirs exist
    Path(os.path.dirname(args.embeddings_out)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.index_out)).mkdir(parents=True, exist_ok=True)

    # Load dataframe
    df = pd.read_csv(args.item_images_csv)
    print(f"[INFO] Loaded {len(df)} rows from {args.item_images_csv}")

    # Compute embeddings
    embeddings = embed_images(
        df,
        device=args.device,
        batch_size=16
    )
    print(f"[INFO] Computed embeddings: {embeddings.shape}")

    # Build FAISS index
    index = build_faiss_index(embeddings, metric="l2")

    # Save artifacts
    np.save(args.embeddings_out, embeddings.astype(np.float32))
    faiss.write_index(index, args.index_out)
    print(f"[INFO] Saved embeddings to {args.embeddings_out}")
    print(f"[INFO] Saved FAISS index to {args.index_out}")

    # Save metadata we’ll need at inference time
    meta = {
        "image_ids": df["image_id"].tolist(),
        "garment_ids": df["garment_id"].tolist(),
        "image_paths": df["image_path"].tolist(),
        "metric": args.metric,
        "model_name": args.model_name,
    }
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved metadata to {args.meta_out}")


if __name__ == "__main__":
    main()