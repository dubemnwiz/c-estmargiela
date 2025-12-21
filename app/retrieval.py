# app/retrieval.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image


@dataclass
class RetrievalResult:
    garment_id: str
    score: float
    support: int  # how many neighbors contributed


class GarmentRetriever:
    """
    Loads:
      - FAISS index (L2)
      - meta.json mapping (parallel arrays: image_ids, garment_ids, image_paths)
      - items.csv for garment_id -> common_name
      - SigLIP model for query embedding
    """
    def __init__(
        self,
        index_path: str,
        meta_path: str,
        items_csv: str,
        device: Optional[str] = None,
        model_name: str = "ViT-B-16-SigLIP",
        pretrained: str = "webli",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load FAISS index (L2)
        self.index = faiss.read_index(index_path)

        # Load meta sidecar
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.image_ids: List[str] = meta["image_ids"]
        self.garment_ids: List[str] = meta["garment_ids"]
        self.image_paths: List[str] = meta["image_paths"]

        if not (len(self.image_ids) == len(self.garment_ids) == len(self.image_paths)):
            raise ValueError("meta.json arrays must be same length")

        # Load garment names (optional but nice)
        self.garment_name: Dict[str, str] = {}
        if Path(items_csv).exists():
            items_df = pd.read_csv(items_csv)
            if "garment_id" in items_df.columns and "common_name" in items_df.columns:
                self.garment_name = dict(zip(items_df["garment_id"], items_df["common_name"]))

        # Load SigLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def embed_pil(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)  # (1, C, H, W)
        feat = self.model.encode_image(x)                      # (1, d)
        feat = feat / feat.norm(dim=-1, keepdim=True)         # L2 normalize
        return feat.cpu().numpy().astype(np.float32)          # (1, d)

    def search_images(self, q: np.ndarray, top_k: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          D: distances (FAISS L2 returns squared L2 distances for IndexFlatL2)
          I: indices into meta arrays
        """
        D, I = self.index.search(q, top_k)
        return D[0], I[0]

    @staticmethod
    def l2_to_cosine_score(dist_sq: float) -> float:
        """
        If vectors are L2-normalized:
          ||a-b||^2 = 2 - 2*cos(a,b)  =>  cos = 1 - dist_sq/2
        This gives a nice similarity-like score in [-1, 1].
        """
        return float(1.0 - (dist_sq / 2.0))

    def predict_garments(
        self,
        img: Image.Image,
        top_k_images: int = 25,
        top_k_garments: int = 5,
    ) -> Dict:
        q = self.embed_pil(img)
        dist_sq, idxs = self.search_images(q, top_k=top_k_images)

        # Per-image hits
        image_hits = []
        for d, i in zip(dist_sq.tolist(), idxs.tolist()):
            if i < 0:
                continue
            gid = self.garment_ids[i]
            image_hits.append({
                "image_id": self.image_ids[i],
                "garment_id": gid,
                "score": self.l2_to_cosine_score(d),
                "image_path": self.image_paths[i],
            })

        # Aggregate per-garment (mean score over supporting neighbors)
        agg_scores: Dict[str, List[float]] = {}
        for hit in image_hits:
            agg_scores.setdefault(hit["garment_id"], []).append(hit["score"])

        garment_ranked = sorted(
            ((gid, float(np.mean(scores)), len(scores)) for gid, scores in agg_scores.items()),
            key=lambda x: x[1],
            reverse=True,
        )

        garment_hits = []
        for gid, s, n in garment_ranked[:top_k_garments]:
            garment_hits.append({
                "garment_id": gid,
                "common_name": self.garment_name.get(gid),
                "score": s,
                "support": n,
            })

        best = garment_hits[0] if garment_hits else None

        return {
            "best": best,
            "top_garments": garment_hits,
            "top_images": image_hits[:min(len(image_hits), top_k_images)],
        }
