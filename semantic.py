# ===========================
# semantic.py — búsqueda semántica con embeddings
# El modelo se encarga de:
# - Cargar el modelo de SentenceTransformers (MiniLM-L6)
# - Convertir productos (texto) en vectores numéricos (embeddings)
# - Construir un índice FAISS para búsquedas rápidas
# - Funciñon semantic_search(query, k) para obtener IDs similares
# ===========================


# --- IMPORTS --------------------------------------------------------------
from pathlib import Path
import os
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer


# --- PARCHE ANTI-CRASH EN MAC (MPS/CUDA) Y LÍMITES DE HILOS ---------------
# si MPS falla, usa CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# ignora CUDA si existiera
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# menos warnings y conflictos de hilos
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# limitar hilos para evitar oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
# para BLAS/NumPy
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# para MKL
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Se limitan los hilos de PyTorch para no saturar el sistema
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# --- CONFIGURACIÓN DEL MODELO Y CARPETA DE TRABAJO ------------------------
# Modelo ligero para búsqueda semántica general
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Carpeta donde se guardan:
# - embeddings.npy: vectores de todos los productos
# - id_map.npy: mapa de índices a IDs reales
# - faiss.index: índice FAISS para búsquedas rápidas
VAR_DIR = Path(__file__).parent / "var"
VAR_DIR.mkdir(exist_ok=True)


# --- OBJETOS GLOBALES (SE CARGAN UNA VEZ) ---------------------------------
# _model: modelo de SentenceTransformers
# _index: índice FAISS en memoria
# _id_map: mapa de índices a IDs reales
_model = None
_index = None
_id_map = None


# --- CARGA DE LOS EMBEDDINGS Y CONSTRUCCIÓN DEL ÍNDICE --------------------
# Carga el modelo de embeddings (si no está cargado ya) y lo deja en memoria
# Se fuerza el uso de la CPU pasra evitar problemas con GPU/MPS en algunos sistemas
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


# --- CONSTRUCCIÓN Y BÚSQUEDA EN EL ÍNDICE SEMÁNTICO -----------------------
# Construye un texto completo por producto combinando: título, descripción y tags
# Devuelve lista de textos.
def build_corpus(df: pd.DataFrame) -> list[str]:
    return (
        df["title"].fillna("") + ". " +
        df["description"].fillna("") + ". " +
        df["tags"].fillna("")
    ).tolist()


# --- EMBEDDING DE TEXTOS ---------------------------------------------------
# Convierte una lista de textos en una matriz de embeddings normalizados
# normalize_embeddings=True para usar similitud coseno directamente con FAISS
# Devuelve un array NumPy de tipo float32
def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype="float32")


# --- GUARDAR Y CARGAR ARTÍCULOS DEL ÍNDICE ---------------------------------
# Guarda en disco:
# - embeddings.npy: matriz de embeddings
# - id_map.npy: mapa de índices a IDs reales
# - faiss.index: índice FAISS
def save_artifacts(index, embs: np.ndarray, id_map: np.ndarray):
    np.save(VAR_DIR / "embeddings.npy", embs)
    np.save(VAR_DIR / "id_map.npy", id_map)
    faiss.write_index(index, str(VAR_DIR / "faiss.index"))

# Intenta cargar los artefactos del índice desde disco generados previamente
def load_artifacts():

    global _index, _id_map

    # Rutas de los archivos
    emb_path = VAR_DIR / "embeddings.npy"
    map_path = VAR_DIR / "id_map.npy"
    idx_path = VAR_DIR / "faiss.index"

    # Si existen, los carga en memoria
    if emb_path.exists() and map_path.exists() and idx_path.exists():
        # cargar desde disco
        _id_map = np.load(map_path)
        # cargar índice FAISS desde disco
        _index = faiss.read_index(str(idx_path))
        return True
    return False


# --- CONSTRUCCIÓN O RECONSTRUCCIÓN DEL ÍNDICE ------------------------------
# Construye el índice FAISS desde el DataFrame completo
def build_or_rebuild_index(df: pd.DataFrame):
    global _index, _id_map

    # Construir corpus
    corpus = build_corpus(df)
    # Obtener embeddings
    embs = embed_texts(corpus)
    # Crear índice FAISS
    dim = embs.shape[1]         

    # Índice de similitud coseno (Inner Product con embs normalizados)
    _index = faiss.IndexFlatIP(dim)
    # Añadir embeddings al índice
    _index.add(embs)
    # Mapa de índices a IDs reales
    _id_map = df["id"].to_numpy()
    # Guardar artefactos en disco
    save_artifacts(_index, embs, _id_map)


# --- FUNCIÓN PRINCIPAL DE BÚSQUEDA SEMÁNTICA -------------------------------
# Asegura que el índice esté cargado o construido
def ensure_index(df: pd.DataFrame):
    # Intenta cargar artefactos; si falla, construye el índice
    if not load_artifacts():
        build_or_rebuild_index(df)

# Búsqueda semántica: devuelve IDs de productos similares a la query
def semantic_search(query: str, k: int = 8) -> list[int]:
    global _index, _id_map

    # Asegura que el índice esté listo
    model = get_model()
    # Si el índice no está cargado, lanza error
    q_emb = model.encode(
        [query], 
        normalize_embeddings=True, 
        show_progress_bar=False
        ).astype("float32")
    # Scores: similitudes e índices de los k más cercanos
    scores, idxs = _index.search(q_emb, k)
    idxs = idxs[0]
    # Convertimos posiciones a IDs reales, ignorando -1 (no encontrados)
    return [_id_map[i] for i in idxs if i != -1]