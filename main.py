# ===========================
# Stussy · E-commerce con IA
# Backend en FastAPI (API + HTML)
# ===========================

# --- IMPORTS (librerías que usamos) ---
from fastapi import FastAPI, Request                 # FastAPI: framework web; Request: objeto de petición HTTP
from fastapi.responses import HTMLResponse           # Para devolver HTML (no solo JSON)
from fastapi.staticfiles import StaticFiles          # Para servir archivos estáticos (CSS, imágenes, JS)
from fastapi.templating import Jinja2Templates       # Motor de plantillas HTML (inyectamos datos en index.html)
from pathlib import Path                             # Manejo de rutas de archivos/carpetas multiplataforma
import pandas as pd                                  # Lectura y manejo de datos tipo tabla (CSV)

from pydantic import BaseModel
from datetime import datetime
from fastapi import Body
from fastapi.responses import JSONResponse

# SEMANTICA
# Módulos de IA clásicos (scikit-learn) para recomendación por contenido
from sklearn.feature_extraction.text import TfidfVectorizer  # Convierte texto → vectores numéricos (TF-IDF)
from sklearn.metrics.pairwise import cosine_similarity        # Calcula similitud entre vectores (0 a 1)

# Con esto podemos construir/cargar el índice FAISS y hacer búsquedas semánticas
from fastapi.responses import HTMLResponse, JSONResponse
from semantic import ensure_index, semantic_search

# Conexion para que la API acepte llamadas desde navegadores externos como ChatGPT
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIGURACIÓN BASE ---
BASE_DIR = Path(__file__).parent
# BASE_DIR es la carpeta donde está este archivo main.py (sirve para construir rutas relativas seguras)

app = FastAPI(title="Stussy · E-commerce IA")
# Creamos la aplicación FastAPI y le damos un título (aparece en /docs)

# Conexion CORS para permitir llamadas desde otros orígenes (ChatGPT, MCP, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las URLs (ajusta esto en producción)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MONTAR CARPETA DE ESTÁTICOS (CSS/IMG/JS) ---
# Todo lo que se pida en /static/... lo servimos desde la carpeta local "static"
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

# --- MOTOR DE PLANTILLAS (HTML con Jinja2) ---
# Le decimos dónde están los .html para renderizarlos
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# SEMANTICA
@app.on_event("startup")
def _startup():
    # Carga o construye el índice semántico una vez al iniciar el server.
    # Sirve para no recalcular en cada petición; deja la IA lista para usar.
    df = load_df()
    ensure_index(df)

# ===========================
# FUNCIONES AUXILIARES
# ===========================
    
# SEMANTICA
def load_df():
    """
    Normaliza tipos(id, precio, stock) y pone imagen por defecto si falta.
    Sirve para evitar errores de formato y sorpresas al buscar o recomendar.
    """

    """
    Construye un 'corpus' de texto por producto combinando título, descripción y tags.
    Este texto es el que usaremos para:
      - Vectorizar con TF-IDF
      - Calcular similitudes entre productos
    """
    df = pd.read_csv(BASE_DIR / "products.csv")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0).astype(int)
    df["image"] = df["image"].fillna("/static/img/placeholder.jpg")
    return df
    

def build_corpus(df: pd.DataFrame) -> list[str]:
    """
    Construye un 'corpus' de texto por producto combinando título, descripción y tags.
    Este texto es el que usaremos para:
      - Vectorizar con TF-IDF
      - Calcular similitudes entre productos
    """
    # fillna("") evita errores si alguna columna tiene valores nulos
    corpus = (
        df["title"].fillna("") + ". " +
        df["description"].fillna("") + ". " +
        df["tags"].fillna("")
    ).tolist()
    return corpus

# ===========================
# RUTAS (ENDPOINTS) DE LA API
# ===========================

@app.get("/health")
def health() -> dict:
    """
    Comprobación rápida de salud del servidor.
    Si responde {"ok": True}, el backend está vivo.
    """
    return {"ok": True}

@app.get("/products")
def list_products() -> list[dict]:
    """
    Devuelve el catálogo completo de productos en formato JSON.
    Útil para debug o para que un frontend consuma la API.
    """
    df = load_df()
    # to_dict(orient="records") → lista de diccionarios [{col1:..., col2:...}, ...]
    return df.to_dict(orient="records")

@app.get("/recommend")
def recommend(product_id: int, k: int = 6) -> list[dict]:
    """
    Recomendador por contenido:
    - product_id: ID del producto base para buscar similares
    - k: número de recomendaciones a devolver (por defecto 6)
    Pasos:
      1) Cargar productos y construir corpus (texto por producto)
      2) Vectorizar corpus con TF-IDF
      3) Calcular similitud del producto base contra todos
      4) Ordenar por similitud y devolver top-k (excluyendo el propio)
    """
    df = load_df()

    # Si el product_id no existe en el CSV, devolvemos lista vacía
    if product_id not in df["id"].values:
        return []

    # 1) Construimos el corpus de texto (uno por producto)
    corpus = build_corpus(df)

    # 2) Vectorizamos con TF-IDF (max_features limita el tamaño del vocabulario para rendimiento)
    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(corpus)  # X es una matriz (n_productos x n_palabras)

    # 3) Encontrar el índice (posición) del producto base dentro del DataFrame
    idx = df.index[df["id"] == product_id][0]

    # 4) Similitud coseno del producto base vs. todos los productos
    #    Resultado: array con un score por cada producto (0 = nada similar, 1 = idéntico)
    sims = cosine_similarity(X[idx], X).ravel()

    # Guardamos la similitud en el DataFrame para poder ordenar y filtrar
    df = df.copy()                # copiamos para no modificar el original
    df["similarity"] = sims

    # Excluimos el mismo producto y ordenamos por similitud descendente
    recs = (
        df[df["id"] != product_id]
        .sort_values("similarity", ascending=False)
        .head(k)
    )

    # Devolvemos solo las columnas que interesan (puedes añadir/quitar a gusto)
    cols = ["id", "title", "description", "price", "stock", "tags", "image", "similarity"]
    return recs[cols].to_dict(orient="records")

@app.get("/search")
def search(q: str, k: int = 8) -> list[dict]:
    """
    Búsqueda simple por palabra clave (baseline):
    - q: texto a buscar (ej. 'sudadera', 'algodón')
    - k: máximo de resultados
    Busca 'q' en title, description o tags (sin IA por ahora).
    *Más adelante reemplazaremos esto por búsqueda semántica con embeddings + FAISS.*
    """
    df = load_df()
    ql = q.lower()

    # Función auxiliar: comprueba si 'q' está dentro de un texto (ignorando mayúsculas)
    def contains(s) -> bool:
        return ql in str(s).lower()

    # Máscara booleana: True donde hay coincidencia en alguna de las columnas
    mask = (
        df["title"].apply(contains) |
        df["description"].apply(contains) |
        df["tags"].apply(contains)
    )

    # Devolvemos hasta k resultados
    return df[mask].head(k).to_dict(orient="records")

@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """
    Página principal (HTML):
    - Carga productos del CSV
    - Renderiza 'templates/index.html' pasándole la lista de productos
    Nota: el parámetro 'request' es obligatorio para Jinja2 en FastAPI.
    """
    df = load_df()
    products = df.to_dict(orient="records")

    # Renderizamos la plantilla index.html con los datos
    # En el HTML, podrás iterar 'products' y pintar el catálogo
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "products": products
        }
    )


# SEMANTICA
@app.get("/semantic-search")
def semantic(q: str, k: int = 8):
    """
    Búsqueda semántica: entiende el significado. una ruta como /search pero con significado real.
    Devuelve productos más cercanos a la query por embeddings + FAISS.
    """
    df = load_df()
    ids = semantic_search(q, k)  # devuelve lista de IDs en orden de relevancia
    res = df[df["id"].isin(ids)].copy()

    # Ordenar los resultados en el mismo orden que FAISS
    order = {pid: i for i, pid in enumerate(ids)}
    res["__ord"] = res["id"].map(order)
    res = res.sort_values("__ord").drop(columns="__ord")

    return JSONResponse(res.to_dict(orient="records"))


class ChatRequest(BaseModel):
    question: str

def log_event(kind: str, payload: dict):
    """Guarda métricas simples en logs/events.csv"""
    import csv
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    path = logs_dir / "events.csv"
    row = {"ts": datetime.now().isoformat(timespec="seconds"), "kind": kind, **payload}
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)

def rank_products_tfidf(query: str, k: int = 4):
    df = load_df()
    corpus = (df["title"].fillna("") + ". " +
              df["description"].fillna("") + ". " +
              df["tags"].fillna(""))
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(corpus.tolist())
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    df = df.copy()
    df["score"] = sims
    return df.sort_values("score", ascending=False).head(k)

def rank_products_semantic(query: str, k: int = 4):
    """Usa semántica si está disponible; si falla, cae a TF-IDF."""
    try:
        from semantic import semantic_search
        df = load_df()
        ids = semantic_search(query, k)
        res = df[df["id"].isin(ids)].copy()
        order = {pid: i for i, pid in enumerate(ids)}
        res["__ord"] = res["id"].map(order)
        return res.sort_values("__ord").drop(columns="__ord")
    except Exception:
        return rank_products_tfidf(query, k)
    

@app.post("/chat")
def chat(req: ChatRequest):
    q = (req.question or "").strip()
    if not q:
        return JSONResponse({
            "answer": "¿En qué puedo ayudarte? Por ejemplo: 'sudadera negra talla M' o 'zapatillas cómodas'.",
            "products": []
        })

    # Log de entrada
    log_event("chat_query", {"q": q})

    # Recupera productos (semántica -> tfidf)
    top = rank_products_semantic(q, k=2)
    prods = top.to_dict(orient="records") if not top.empty else []

    # Redacta respuesta simple
    if not prods:
        answer = "No he encontrado productos para esa búsqueda. Prueba con 'sudadera', 'zapatillas' o 'gorra'."
    else:
        nombres = ", ".join([p["title"] for p in prods[:3]])
        answer = f"Te pueden encajar: {nombres}. ¿Quieres que filtre por precio o por stock?"

    # Log de salida
    log_event("chat_response", {"q": q, "n": len(prods)})

    # Devuelve productos con campos clave
    cols = ["id","title","description","price","stock","tags","image"]
    items = [{c: p.get(c) for c in cols} for p in prods]

    return JSONResponse({"answer": answer, "products": items})


def apply_structured_filters(df, category=None, max_price=None, in_stock=None):
    """Filtra por categoría, precio y stock sobre un DataFrame de productos."""
    out = df.copy()

    # Categoría (busca en title, description y tags)
    if category:
        cat = str(category).lower()
        mask = (
            out["title"].str.contains(cat, case=False, na=False) |
            out["description"].str.contains(cat, case=False, na=False) |
            out["tags"].astype(str).str.contains(cat, case=False, na=False)
        )
        out = out[mask]

    # Precio máximo
    if max_price is not None:
        out["__price"] = pd.to_numeric(out["price"], errors="coerce")
        out = out[out["__price"] <= float(max_price)]

    # Solo con stock
    if in_stock is True:
        out["__stock"] = pd.to_numeric(out["stock"], errors="coerce")
        out = out[out["__stock"] > 0]

    # Limpieza de columnas temporales si existen
    for col in ["__price", "__stock"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out

@app.post("/agent")
def agent(req: dict = Body(...)):
    """
    Endpoint universal para agentes (ChatGPT, MCP, etc.)
    Acciones:
      - search:    query + (category?, max_price?, in_stock?)
      - recommend: product_id
      - chat:      query
    """
    action     = req.get("action")
    query      = req.get("query", "")
    product_id = req.get("product_id")

    # Filtros estructurados (opcional en search)
    category   = req.get("category")        # ej: "sudadera"
    max_price  = req.get("max_price")       # ej: 70
    in_stock   = req.get("in_stock")        # True/False

    # 1) BÚSQUEDA
    if action == "search" and (query or category or max_price is not None or in_stock is not None):
        # Traemos bastantes candidatos y luego filtramos
        try:
            top = rank_products_semantic(query or "", k=50)
        except Exception:
            top = rank_products_tfidf(query or "", k=50)

        top = apply_structured_filters(top, category=category, max_price=max_price, in_stock=in_stock)
        top = top.head(3)  # limita respuesta final

        return {
            "source": "stussy-ai",
            "type": "search_results",
            "query": query,
            "filters": {"category": category, "max_price": max_price, "in_stock": in_stock},
            "results": top.to_dict(orient="records")
        }

    # 2) RECOMENDACIONES POR ID
    if action == "recommend" and product_id is not None:
        recs = recommend(int(product_id))  # ya tienes esta función
        # aplica filtros si los mandan (opcional)
        df = pd.DataFrame(recs)
        if not df.empty:
            df = apply_structured_filters(df, category=category, max_price=max_price, in_stock=in_stock)
            recs = df.head(3).to_dict(orient="records")
        return {
            "source": "stussy-ai",
            "type": "recommendations",
            "product_id": product_id,
            "filters": {"category": category, "max_price": max_price, "in_stock": in_stock},
            "results": recs
        }

    # 3) CHAT (reutiliza tu /chat)
    if action == "chat" and query:
        msg = chat(ChatRequest(question=query))
        return {"source": "stussy-ai", "type": "chat_response", "data": msg}

    return {"error": "Acción no reconocida o parámetros incompletos."}
