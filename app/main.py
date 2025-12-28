# app/main.py
import os

# If your Mac is finicky, these help stability (you already used them in shell).
# Keeping them here prevents “forgot to export” issues.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from io import BytesIO
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from PIL import Image

from app.retrieval import GarmentRetriever

app = FastAPI(title="C’est Margiela", version="0.2.0")

# ---- Load once at startup ----
RETRIEVER: GarmentRetriever | None = None

INDEX_PATH = "data/index/margiela_items_siglip.faiss"
META_PATH = "data/index/margiela_items_siglip_meta.json"
ITEMS_CSV = "data/catalog/margiela/items.csv"


@app.on_event("startup")
def _startup():
    global RETRIEVER
    RETRIEVER = GarmentRetriever(
        index_path=INDEX_PATH,
        meta_path=META_PATH,
        items_csv=ITEMS_CSV,
        model_name="ViT-B-16-SigLIP",
        pretrained="webli",
    )


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": RETRIEVER is not None}


@app.post("/v1/query")
async def query(
    mode: str = Query("garment"),
    file: UploadFile = File(...),
    top_k_images: int = Query(25, ge=1, le=200),
    top_k_garments: int = Query(5, ge=1, le=50),
    debug: bool = Query(False, description="If true, include top_images + per-crop breakdown"),
    unknown_threshold: float = Query(0.55, ge=-1.0, le=1.0),
    unknown_margin: float = Query(0.05, ge=0.0, le=2.0),
    exclude_image_path: str | None = Query(
        None,
        description="Optional: exclude a specific catalog image_path (useful for local self-match testing)",
    ),
):
    if mode != "garment":
        raise HTTPException(status_code=400, detail="Only mode=garment supported right now")

    if RETRIEVER is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        img = Image.open(BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = RETRIEVER.predict_garments(
        img=img,
        top_k_images=top_k_images,
        top_k_garments=top_k_garments,
        debug=debug,
        unknown_threshold=unknown_threshold,
        unknown_margin=unknown_margin,
        exclude_image_path=exclude_image_path,
    )

    return {
        "mode": "garment",
        "filename": file.filename,
        **result,
    }
