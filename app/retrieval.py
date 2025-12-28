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
    
    # ---------- Crop V1 helpers ----------
    @staticmethod
    def _crop_full(img: Image.Image) -> Image.Image:
        return img

    @staticmethod
    def _crop_upper(img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.crop((0, 0, w, int(h * 0.60)))

    @staticmethod
    def _crop_lower(img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.crop((0, int(h * 0.45), w, h))

    def _predict_one_crop(
        self,
        img: Image.Image,
        top_k_images: int,
        exclude_image_path: Optional[str] = None,
        crop_tag: str = "full",
    ) -> Dict:
        q = self.embed_pil(img)
        dist_sq, idxs = self.search_images(q, top_k=top_k_images)

        image_hits = []
        for d, i in zip(dist_sq.tolist(), idxs.tolist()):
            if i < 0:
                continue
            path = self.image_paths[i]
            if exclude_image_path and path == exclude_image_path:
                continue

            image_hits.append(
                {
                    "image_id": self.image_ids[i],
                    "garment_id": self.garment_ids[i],
                    "score": self.l2_to_cosine_score(d),
                    "image_path": path,
                    "crop": crop_tag,
                }
            )

        # Aggregate per garment: mean of contributing image scores
        agg_scores: Dict[str, List[float]] = {}
        for hit in image_hits:
            agg_scores.setdefault(hit["garment_id"], []).append(hit["score"])

        garment_hits = sorted(
            (
                {
                    "garment_id": gid,
                    "common_name": self.garment_name.get(gid),
                    "score": float(np.mean(scores)),
                    "support": int(len(scores)),
                    "crop": crop_tag,
                }
                for gid, scores in agg_scores.items()
            ),
            key=lambda x: x["score"],
            reverse=True,
        )

        best = garment_hits[0] if garment_hits else None
        return {"best": best, "top_garments": garment_hits, "top_images": image_hits}


    def predict_garments(
        self,
        img: Image.Image,
        top_k_images: int = 25,
        top_k_garments: int = 5,
        debug: bool = False,
        # Unknown logic
        unknown_threshold: float = 0.55,
        unknown_margin: float = 0.05,
        # Local testing convenience: exclude exact catalog path if you know it
        exclude_image_path: Optional[str] = None,
    ) -> Dict:
        """
        Crop V1:
          - full, upper, lower
        Merge strategy (simple + effective):
          - per garment: take MAX score across crops
          - support: sum supports across crops
          - track which crop produced max score
        Unknown:
          - if best.score < threshold OR margin < unknown_margin -> unknown
        Debug:
          - include top_images + per-crop breakdown
        """
        img = img.convert("RGB")

        crops = [
            ("full", self._crop_full(img)),
            ("upper", self._crop_upper(img)),
            ("lower", self._crop_lower(img)),
        ]

        per_crop = {}
        for tag, cimg in crops:
            per_crop[tag] = self._predict_one_crop(
                cimg, top_k_images=top_k_images, exclude_image_path=exclude_image_path, crop_tag=tag
            )

        # Merge garment scores across crops
        merged: Dict[str, Dict] = {}
        for tag in per_crop:
            for g in per_crop[tag]["top_garments"]:
                gid = g["garment_id"]
                if gid not in merged:
                    merged[gid] = {
                        "garment_id": gid,
                        "common_name": self.garment_name.get(gid),
                        "score": g["score"],
                        "support": g["support"],
                        "best_crop": tag,
                        "per_crop": {tag: {"score": g["score"], "support": g["support"]}},
                    }
                else:
                    # keep max score across crops
                    if g["score"] > merged[gid]["score"]:
                        merged[gid]["score"] = g["score"]
                        merged[gid]["best_crop"] = tag
                    merged[gid]["support"] += g["support"]
                    merged[gid]["per_crop"][tag] = {"score": g["score"], "support": g["support"]}

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        top_ranked = ranked[:top_k_garments]

        best = top_ranked[0] if top_ranked else None
        second = top_ranked[1] if len(top_ranked) > 1 else None
        margin = (best["score"] - second["score"]) if (best and second) else None

        is_unknown = False
        reason = None
        if best is None:
            is_unknown = True
            reason = "no_candidates"
        else:
            if best["score"] < unknown_threshold:
                is_unknown = True
                reason = "below_threshold"
            elif margin is not None and margin < unknown_margin:
                is_unknown = True
                reason = "low_margin"

        best_out = None
        if is_unknown:
            best_out = {
                "garment_id": "unknown",
                "common_name": None,
                "score": float(best["score"]) if best else None,
                "support": int(best["support"]) if best else 0,
                "best_crop": best["best_crop"] if best else None,
                "reason": reason,
                "margin": float(margin) if margin is not None else None,
            }
        else:
            best_out = {
                "garment_id": best["garment_id"],
                "common_name": best["common_name"],
                "score": float(best["score"]),
                "support": int(best["support"]),
                "best_crop": best["best_crop"],
                "margin": float(margin) if margin is not None else None,
            }

        response = {
            "best": best_out,
            "top_garments": [
                {
                    "garment_id": g["garment_id"],
                    "common_name": g["common_name"],
                    "score": float(g["score"]),
                    "support": int(g["support"]),
                    "best_crop": g["best_crop"],
                }
                for g in top_ranked
            ],
            "meta": {
                "unknown_threshold": float(unknown_threshold),
                "unknown_margin": float(unknown_margin),
                "margin": float(margin) if margin is not None else None,
                "is_unknown": bool(is_unknown),
                "unknown_reason": reason,
            },
        }

        if debug:
            # include per-crop top images + per-crop top garments
            response["debug"] = {
                "per_crop": {
                    tag: {
                        "best": per_crop[tag]["best"],
                        "top_garments": per_crop[tag]["top_garments"][:top_k_garments],
                        "top_images": per_crop[tag]["top_images"][:top_k_images],
                    }
                    for tag in per_crop
                }
            }

        return response