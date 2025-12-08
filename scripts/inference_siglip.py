#!/usr/bin/env python
#!/usr/bin/env python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image


DEFAULT_INDEX_PATH = "data/index/margiela_items_siglip.faiss"
DEFAULT_META_PATH = "data/index/margiela_items_siglip_meta.json"


def load_siglip(
    model_name: str = "ViT-B-16-SigLIP",
    pretrained: str = "webli",
    device: str = "cpu",
):
    """
    Load SigLIP via OpenCLIP's built-in weights.

    Must match the model used in build_item_index_siglip.py
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    model.eval()
    model.to(device)
    return model, preprocess


class RetrievalEngine:
    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        meta_path: str = DEFAULT_META_PATH,
        device: str = None,
    ):
        """
        index_path: path to FAISS index (.faiss)
        meta_path: path to metadata JSON produced by build_item_index_siglip.py
        """
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 1) Load metadata (image_ids, garment_ids, etc.)
        with open(self.meta_path, "r") as f:
            meta = json.load(f)

        self.image_ids: List[str] = meta["image_ids"]
        self.garment_ids: List[str] = meta["garment_ids"]
        self.image_paths: List[str] = meta.get("image_paths", [])
        self.metric: str = meta.get("metric", "ip")
        self.model_name: str = meta.get("model_name", "ViT-B-16-SigLIP")
        self.pretrained_tag: str = meta.get("pretrained", "webli")

        # 2) Load FAISS index
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        if self.index.ntotal != len(self.image_ids):
            print(
                f"[WARN] index vectors ({self.index.ntotal}) "
                f"!= number of image_ids ({len(self.image_ids)})"
            )

        print(
            f"[inference] Loaded FAISS index with {self.index.ntotal} vectors "
            f"from {self.index_path}"
        )

        # 3) Load SigLIP model + preprocess (must match build script)
        print(
            f"[inference] Loading SigLIP model '{self.model_name}' "
            f"pretrained='{self.pretrained_tag}' on {self.device}"
        )
        self.model, self.preprocess = load_siglip(
            model_name=self.model_name,
            pretrained=self.pretrained_tag,
            device=self.device,
        )

    # ---------- internal helpers ----------

    def _embed_image(self, img: Image.Image) -> np.ndarray:
        """
        Embed a single PIL image into a normalized SigLIP vector (1, D).
        """
        self.model.eval()
        with torch.no_grad():
            x = self.preprocess(img).unsqueeze(0).to(self.device)
            feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
        return feats.cpu().numpy().astype("float32")  # shape: (1, D)

    def _faiss_search(
        self,
        query_emb: np.ndarray,
        top_k_images: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run FAISS search.

        query_emb: (1, D)
        returns:
            scores_or_dists: (top_k_images,)
            idxs: (top_k_images,)
        """
        scores, idxs = self.index.search(query_emb, top_k_images)
        return scores[0], idxs[0]

    def _convert_to_scores(self, vals: np.ndarray) -> np.ndarray:
        """
        Convert FAISS search outputs into "scores" where higher is better.

        - For inner product ('ip'), FAISS already returns similarities.
        - For L2 ('l2'), FAISS returns distances (smaller is better), so we
          convert to scores by negating.
        """
        if self.metric.lower() == "ip":
            return vals  # already similarity
        elif self.metric.lower() == "l2":
            return -vals  # smaller distance -> larger score
        else:
            # default: treat as similarity
            return vals

    def _aggregate_to_garments(
        self,
        scores: np.ndarray,
        idxs: np.ndarray,
        top_k_garments: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Aggregate image-level scores into garment-level scores.

        Simple strategy: sum scores for each garment across top image hits.
        """
        garment_scores: Dict[str, float] = {}

        for score, idx in zip(scores, idxs):
            gid = self.garment_ids[int(idx)]
            garment_scores[gid] = garment_scores.get(gid, 0.0) + float(score)

        # sort garments by aggregated score (desc)
        ranked = sorted(garment_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k_garments]

    # ---------- public API ----------

    def retrieve_garment(
        self,
        image_path: str,
        top_k_images: int = 20,
        top_k_garments: int = 5,
    ) -> Dict[str, Any]:
        """
        Main inference function.

        Steps:
        - Load query image from disk
        - Embed with SigLIP
        - FAISS search over catalog index
        - Convert FAISS outputs to scores
        - Aggregate scores per garment_id
        - Return best garment + ranked list + raw neighbors
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Query image not found: {path}")

        img = Image.open(path).convert("RGB")

        # 1) embed query
        query_emb = self._embed_image(img)

        # 2) FAISS search
        vals, idxs = self._faiss_search(query_emb, top_k_images=top_k_images)

        # 3) convert to scores (higher is better)
        scores = self._convert_to_scores(vals)

        # 4) aggregate to garment-level scores
        garment_rankings = self._aggregate_to_garments(
            scores, idxs, top_k_garments=top_k_garments
        )

        if not garment_rankings:
            raise RuntimeError("No garments returned from aggregation.")

        best_garment_id, best_score = garment_rankings[0]

        # 5) raw image neighbors for debugging
        raw_neighbors = []
        for val, score, idx in zip(vals, scores, idxs):
            idx = int(idx)
            raw_neighbors.append(
                {
                    "image_id": self.image_ids[idx],
                    "garment_id": self.garment_ids[idx],
                    "image_path": self.image_paths[idx] if self.image_paths else None,
                    "faiss_value": float(val),   # raw FAISS output (sim or dist)
                    "score": float(score),       # normalized so higher is better
                }
            )

        return {
            "best_garment_id": best_garment_id,
            "best_score": float(best_score),
            "metric": self.metric,
            "top_garments": [
                {"garment_id": gid, "score": float(score)}
                for gid, score in garment_rankings
            ],
            "raw_neighbors": raw_neighbors,
        }


# --------- CLI for quick testing ---------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
        help="Path to query image.",
    )
    parser.add_argument(
        "--index-path",
        default=DEFAULT_INDEX_PATH,
        help="Path to FAISS index (.faiss) built by build_item_index_siglip.py",
    )
    parser.add_argument(
        "--meta-path",
        default=DEFAULT_META_PATH,
        help="Path to metadata JSON built by build_item_index_siglip.py",
    )
    parser.add_argument("--top-k-images", type=int, default=20)
    parser.add_argument("--top-k-garments", type=int, default=5)
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cuda | cpu (defaults to cuda if available else cpu)",
    )
    args = parser.parse_args()

    engine = RetrievalEngine(
        index_path=args.index_path,
        meta_path=args.meta_path,
        device=args.device,
    )

    result = engine.retrieve_garment(
        image_path=args.image,
        top_k_images=args.top_k_images,
        top_k_garments=args.top_k_garments,
    )

    import pprint
    pprint.pprint(result)


if __name__ == "__main__":
    main()
