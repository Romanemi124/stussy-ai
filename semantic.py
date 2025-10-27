# semantic.py — búsqueda semántica con embeddings y FAISS (estable en macOS)

from pathlib import Path
import os
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer

# --- Parche anti-crash en Mac (MPS/CUDA) y límites de hilos ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"   # si MPS falla, usa CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""           # ignora CUDA si existiera
os.environ["TOKENIZERS_PARALLELISM"] = "false"    # menos warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Modelo ligero y rápido para e-commerce
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Carpeta para artefactos del índice
VAR_DIR = Path(__file__).parent / "var"
VAR_DIR.mkdir(exist_ok=True)

# Objetos globales (se cargan una vez)
_model = None
_index = None
_id_map = None

def get_model():
    """Carga el modelo de embeddings solo una vez (forzado a CPU)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")  # <- ¡importante!
    return _model

def build_corpus(df: pd.DataFrame) -> list[str]:
    """Texto por producto: título + descripción + tags."""
    return (
        df["title"].fillna("") + ". " +
        df["description"].fillna("") + ". " +
        df["tags"].fillna("")
    ).tolist()

def embed_texts(texts: list[str]) -> np.ndarray:
    """Convierte textos en vectores normalizados (para coseno)."""
    model = get_model()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype="float32")

def save_artifacts(index, embs: np.ndarray, id_map: np.ndarray):
    """Guarda embeddings, mapa de IDs y el índice FAISS."""
    np.save(VAR_DIR / "embeddings.npy", embs)
    np.save(VAR_DIR / "id_map.npy", id_map)
    faiss.write_index(index, str(VAR_DIR / "faiss.index"))

def load_artifacts():
    """Intenta cargar el índice ya construido (arranque rápido)."""
    global _index, _id_map
    emb_path = VAR_DIR / "embeddings.npy"
    map_path = VAR_DIR / "id_map.npy"
    idx_path = VAR_DIR / "faiss.index"
    if emb_path.exists() and map_path.exists() and idx_path.exists():
        _id_map = np.load(map_path)
        _index = faiss.read_index(str(idx_path))
        return True
    return False

def build_or_rebuild_index(df: pd.DataFrame):
    """Construye el índice desde cero a partir del CSV/DF."""
    global _index, _id_map
    corpus = build_corpus(df)
    embs = embed_texts(corpus)
    dim = embs.shape[1]                 # 384 en MiniLM-L6
    _index = faiss.IndexFlatIP(dim)     # Inner Product (coseno con embs normalizados)
    _index.add(embs)
    _id_map = df["id"].to_numpy()
    save_artifacts(_index, embs, _id_map)

def ensure_index(df: pd.DataFrame):
    """Carga índice si existe; si no, lo construye."""
    if not load_artifacts():
        build_or_rebuild_index(df)

def semantic_search(query: str, k: int = 8) -> list[int]:
    """Devuelve IDs de productos más cercanos semánticamente a la query."""
    global _index, _id_map
    model = get_model()
    q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype("float32")
    scores, idxs = _index.search(q_emb, k)
    idxs = idxs[0]
    return [_id_map[i] for i in idxs if i != -1]