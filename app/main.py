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

app = FastAPI(title="C’est Margiela", version="0.1.0")

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
    mode: str = Query("garment", description="Only 'garment' supported in MVP"),
    file: UploadFile = File(...),
    top_k_images: int = Query(25, ge=1, le=200),
    top_k_garments: int = Query(5, ge=1, le=50),
):
    if mode != "garment":
        raise HTTPException(status_code=400, detail="Only mode=garment supported right now")

    if RETRIEVER is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")

    # Read upload
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    # Load as PIL
    try:
        img = Image.open(BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Predict
    result = RETRIEVER.predict_garments(
        img=img,
        top_k_images=top_k_images,
        top_k_garments=top_k_garments,
    )

    return {
        "mode": "garment",
        "filename": file.filename,
        "best": result["best"],
        "top_garments": result["top_garments"],
        "top_images": result["top_images"],
    }
