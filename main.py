# ===========================
# STUSSY AI
# ===========================

# Manejo de rutas de archivos/carpetas multiplataforma
from pathlib import Path
# Para registrar fechas y horas en logs
from datetime import datetime


# --- IMPORTS PARA FASTAPI Y ENTORNO WEB -----------------------------------
# Núcleo FastAPI y tipos para peticiones y respuestas
from fastapi import FastAPI, Request, Body
# Para devolver HTML (no solo JSON)
from fastapi.responses import HTMLResponse, JSONResponse
# Para servir archivos estáticos (CSS, imágenes, JS)
from fastapi.staticfiles import StaticFiles
# Motor de plantillas HTML (inyectamos datos en index.html)
from fastapi.templating import Jinja2Templates
# Conexion para que la API acepte llamadas desde navegadores externos como ChatGPT
from fastapi.middleware.cors import CORSMiddleware

# Para leer y manejar datos (CSV de productos)
import pandas as pd


# --- IMPORTS PARA AI PARA RECOMENDACIONES POR CONTENIDO -------------------
# Convierte texto a vectores numéricos (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
# Calcula similitud entre vectores (0 = nada, 1 = idéntico)
from sklearn.metrics.pairwise import cosine_similarity

# Para definir modelos de datos entrada-salida en FastAPI
from pydantic import BaseModel


# --- IMPORTS PARA BÚSQUEDA SEMÁNTICA CON EMBEDDINGS + FAISS ---------------
# Con esto intentamos importar las funciones semánticas que hemos definido en semantic.py
try:
    from semantic import ensure_index, semantic_search
    SEMANTIC_OK = True
except Exception:
    # Si falla la importación, desactivamos la semántica:
    SEMANTIC_OK = False
    print("Aviso: la búsqueda semántica no está disponible. Revisa semantic.py y las dependencias de IA.")

    def ensure_index(_df):
        return
    
    def semantic_search(query: str, k: int = 8):
        #devuelve lista vacía si no está disponible
        return []


# --- CONFIGURACIÓN BASE DE LA APLICACIÓN ----------------------------------
# BASE_DIR es la carpeta donde está este archivo main.py (sirve para construir rutas relativas seguras)
BASE_DIR = Path(__file__).parent
# Ruta al csv donde se encuentran los productos
CSV_PATH = BASE_DIR / "products.csv"

# Creamos la aplicación FastAPI y le damos un título (aparece en /docs)
app = FastAPI(title="Stussy AI")


# --- CONFIGURACIÓN CORS ---------------------------------------------------
# Esta configuración permite que la API sea llamada desde otros orígenes (en este ejemplo, ChatGPT, MCP, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost:8000",
        "http://127.0.0.1:8000",
        "https://chatgpt.com",
        "https://www.chatgpt.com",
        "*", # Aquí se pueden incluir otros orígenes permitidos
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- ARCHIVOS ESTÁTICOS Y PLANTILLAS HTML ---------------------------------
# Montamos la carpeta "static" para servir CSS, imágenes y JS en la ruta /static
import os
STATIC_PATH = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")


# --- MOTOR DE PLANTILLAS (HTML con Jinja2) --------------------------------
# Configuramos Jinja2 para que cargue las plantillas HTML desde la carpeta "templates"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# --- INICIALIZACIÓN DE ARRANQUE -------------------------------------------
@app.on_event("startup")
def _startup():
    # Carga o construye el índice semántico una vez al iniciar el server.
    # Sirve para no recalcular en cada petición; deja la IA lista para usar.
    if SEMANTIC_OK:
        df = load_df()
        ensure_index(df)


# --- FUNCIONES AUXILIARES DE DATOS ----------------------------------------
def load_df() -> pd.DataFrame:
    # Carga el CSV de productos en un DataFrame de pandas para evitar errores
    df = pd.read_csv(CSV_PATH)
    # Normalizamos el tipo de id (entero)
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    # Normalizamos precio y stock a numérico
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0).astype(int)
    # Si falta imagen, ponemos un placeholder genérico
    df["image"] = df["image"].fillna("/static/img/placeholder.jpg")
    return df
    
# Construye el corpus de texto por producto (titulo, descripción, tags)
# Esto se usa para vectorizar con TF-IDF y calcular similitudes
def build_corpus(df: pd.DataFrame) -> list[str]:
    # Con("") evitamos errores si alguna columna tiene valores nulos
    corpus = (
        df["title"].fillna("") + ". " +
        df["description"].fillna("") + ". " +
        df["tags"].fillna("")
    ).tolist()
    return corpus

# Convierte la URL de imagen de un producto en absoluta si es relativa.
# Ejemplo: "/static/..." → "http://miweb.com/static/..."
# Usamos esto para que los clientes (frontends, ChatGPT, etc.) puedan cargar las imágenes correctamente.
def absolute_image(prod: dict, base: str) -> str:
    img = prod.get("image")
    if img and img.startswith("/"):
        prod["image"] = f"{base}{img}"
    return prod

# Aplica absolute_image a una lista de productos
def absollute_list(items: list[dict], base: str) -> list[dict]:
    for p in items:
        absolute_image(p, base)
    return items


# --- RUTAS (ENDPOINTS) DE LA API ------------------------------------------
@app.get("/health")
# Sirve para comprobar si el servidor está vivo
# Responde {"ok": True} si todo va bien, se usa para monitorizar el estado del backend
def health() -> dict:
    return {"ok": True}

# ---------------------------------------------
# Devuelve el catálogo completo de productos en JSON
@app.get("/products")
def list_products() -> list[dict]:
    df = load_df()
    # to_dict(orient="records"), lista de diccionarios [{col1:..., col2:...}, ...]
    return df.to_dict(orient="records")

# ---------------------------------------------
# Recomendador por contenido (TF-IDF)
@app.get("/recommend")
def recommend(product_id: int, k: int = 6) -> list[dict]:
    
    # product_id: ID del producto base para buscar similares
    # k: número de recomendaciones a devolver (por defecto 6)
    """
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
    # Primero copiamos el DataFrame original para no modificarlo
    df = df.copy()
    df["similarity"] = sims

    # Excluimos el mismo producto y nos quedamos con los más similares
    recs = (
        df[df["id"] != product_id]
        .sort_values("similarity", ascending=False)
        .head(k)
    )

    # Devolvemos solo las columnas que interesan (están puesto todas pero se puede modificar)
    cols = ["id", "title", "description", "price", "stock", "tags", "image", "similarity"]
    return recs[cols].to_dict(orient="records")

# ---------------------------------------------
# Búsqueda simple por palabra clave (baseline)
@app.get("/search")
def search(q: str, k: int = 8) -> list[dict]:

    # q: texto a buscar (ej. 'sudadera', 'algodón')
    # k: máximo de resultados a devolver
    df = load_df()
    ql = q.lower()

    # Función auxiliar: comprueba si 'q' está dentro de un texto (ignorando mayúsculas)
    def contains(s) -> bool:
        return ql in str(s).lower()

    # Creamos una máscara: True donde hay coincidencia en alguna de las columnas
    mask = (
        df["title"].apply(contains) |
        df["description"].apply(contains) |
        df["tags"].apply(contains)
    )

    # Devolvemos hasta k resultados
    return df[mask].head(k).to_dict(orient="records")

# ---------------------------------------------
# Página principal (HTML)
@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:

    # Carga productos del CSV
    # Renderiza 'templates/index.html' pasándole la lista de productos
    # El parámetro 'request' es obligatorio para Jinja2 en FastAPI.
    df = load_df()
    products = df.to_dict(orient="records")

    # Renderizamos la plantilla index.html con los datos, en el HTML, se itera 'products' y pinta el catálogo
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "products": products
        }
    )


# --- BÚSQUEDA SEMÁNTICA ---------------------------------------------------
@app.get("/semantic-search")
#Se activa si semantic.py está activo, devuelve los productos más cercanos a la query
def semantic(q: str, k: int = 8):

    df = load_df()
    # devuelve lista de IDs en orden de relevancia
    ids = semantic_search(q, k)
    res = df[df["id"].isin(ids)].copy()

    # Ordenar los resultados en el mismo orden que devuelve semantic_search
    order = {pid: i for i, pid in enumerate(ids)}
    res["__ord"] = res["id"].map(order)
    res = res.sort_values("__ord").drop(columns="__ord")

    return JSONResponse(res.to_dict(orient="records"))


# --- CHAT SENCILLO DE LA TIENDA -------------------------------------------
# Modelo de datos para la petición de chat (/chat), por ejemplo:
# {
#   "question": "¿Tienes sudaderas negras talla M?"
# }
class ChatRequest(BaseModel):
    question: str

# Función para registrar eventos en logs/events.csv
def log_event(kind: str, payload: dict):
    
    import csv
    logs_dir = BASE_DIR / "logs"
    # Crea la carpeta si no existe
    logs_dir.mkdir(exist_ok=True)
    # Ruta al archivo de logs
    path = logs_dir / "events.csv"
    # Fila a escribir (en formato diccionario) 
    row = {"ts": datetime.now().isoformat(timespec="seconds"), "kind": kind, **payload}
    # Escribe la fila, creando el archivo si no existe
    write_header = not path.exists()
    # Abre en modo append y escribe la fila
    with path.open("a", newline="", encoding="utf-8") as f:
        # Usa csv.DictWriter para escribir el diccionario como fila en el CSV
        w = csv.DictWriter(f, fieldnames=row.keys())
        # Si el archivo es nuevo, escribe la cabecera primero 
        if write_header:
            # Escribe la cabecera
            w.writeheader()
            # Escribe la fila de datos
            w.writerow(row)

# Una vez recibe una pregunta, devuelve los k productos más relevantes usando TF-IDF
def rank_products_tfidf(query: str, k: int = 4):

    # Carga productos y construye corpus
    df = load_df()
    # Solo texto (título + descripción + tags)
    corpus = (df["title"].fillna("") + ". " +
              df["description"].fillna("") + ". " +
              df["tags"].fillna(""))
    
    # Vectoriza con TF-IDF
    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(corpus.tolist())
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    df = df.copy()
    df["score"] = sims
    return df.sort_values("score", ascending=False).head(k)

# Usa semántica si está disponible; si falla, cae a TF-IDF.
def rank_products_semantic(query: str, k: int = 4):
    try:
        # Importa la función semantic_search desde semantic.py
        from semantic import semantic_search
        # Carga productos
        df = load_df()
        # Busca IDs semánticamente similares
        ids = semantic_search(query, k)
        # Filtra DataFrame por esos IDs
        res = df[df["id"].isin(ids)].copy()
        # Ordenar los resultados en el mismo orden que devuelve semantic_search
        order = {pid: i for i, pid in enumerate(ids)}
        # Añade columna temporal para ordenar
        res["__ord"] = res["id"].map(order)
        return res.sort_values("__ord").drop(columns="__ord")
    except Exception:
        return rank_products_tfidf(query, k)
    
# ---------------------------------------------
# Endpoint de chat simple, no es ChatGPT pero sirve para pruebas básicas
@app.post("/chat")
def chat(req: ChatRequest):

    # Extrae y limpia la pregunta
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
    # Lista de productos
    prods = top.to_dict(orient="records") if not top.empty else []

    # Redacta respuesta simple
    if not prods:
        answer = "No he encontrado productos para esa búsqueda. Prueba con 'sudadera', 'zapatillas' o 'gorra'."
    else:
        # Muestra hasta 3 nombres de productos encontrados
        nombres = ", ".join([p["title"] for p in prods[:3]])
        answer = f"Te pueden encajar: {nombres}. ¿Quieres que filtre por precio o por stock?"

    # Log de respuesta
    log_event("chat_response", {"q": q, "n": len(prods)})

    # Devuelve productos con campos clave
    cols = ["id","title","description","price","stock","tags","image"]
    items = [{c: p.get(c) for c in cols} for p in prods]

    # Convierte imágenes a URLs absolutas
    return JSONResponse({"answer": answer, "products": items})


# --- ENDPOINT UNIVERSAL PARA AGENTES (ChatGPT, MCP, etc.) -----------------
def apply_structured_filters(df, category=None, max_price=None, in_stock=None):
    
    # Aplica filtros estructurados a un DataFrame de productos
    # category: filtra por categoría (busca en title, description, tags)
    # max_price: filtra por precio máximo
    # in_stock: si es True, filtra solo productos con stock > 0
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

    # Filtrado por precio máximo
    if max_price is not None:
        out["__price"] = pd.to_numeric(out["price"], errors="coerce")
        out = out[out["__price"] <= float(max_price)]

    # Filtrado por stock disponible
    if in_stock is True:
        out["__stock"] = pd.to_numeric(out["stock"], errors="coerce")
        out = out[out["__stock"] > 0]

    # Elimina columnas temporales usadas para filtrar
    for col in ["__price", "__stock"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out

# ---------------------------------------------
# Endpoint universal para agentes (ChatGPT, MCP, etc.)
@app.post("/agent")
# Recibe peticiones con acción y parámetros variados
def agent(req: dict = Body(...)):
    
    # Extrae parámetros de la petición
    action     = req.get("action")
    query      = req.get("query", "")
    product_id = req.get("product_id")

    # Filtros estructurados (opcional en search)
    category   = req.get("category")        # ej: "sudadera"
    max_price  = req.get("max_price")       # ej: 70
    in_stock   = req.get("in_stock")        # True/False

    # 1) BÚSQUEDA: usa semántica si está disponible, si no TF-IDF + filtros estructurados
    if action == "search" and (query or category or max_price is not None or in_stock is not None):
        # Traemos bastantes candidatos y luego filtramos
        try:
            top = rank_products_semantic(query or "", k=50)
        except Exception:
            top = rank_products_tfidf(query or "", k=50)

        top = apply_structured_filters(top, category=category, max_price=max_price, in_stock=in_stock)
        top = top.head(3)

        return {
            "source": "stussy-ai",
            "type": "search_results",
            "query": query,
            "filters": {"category": category, "max_price": max_price, "in_stock": in_stock},
            "results": top.to_dict(orient="records")
        }

    # 2) RECOMENDACIONES POR ID
    if action == "recommend" and product_id is not None:
        # Usamos esta función ya definida arriba
        recs = recommend(int(product_id))
        # aplica filtros si los mandan (opcional)
        df = pd.DataFrame(recs)

        # Si df no está vacío, aplica filtros
        if not df.empty:
            df = apply_structured_filters(df, category=category, max_price=max_price, in_stock=in_stock)
            # Limita a 3 resultados
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

    # Si no se reconoce la acción, devuelve error
    return {"error": "Acción no reconocida o parámetros incompletos."}
