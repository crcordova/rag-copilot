from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from pandas import DataFrame
from promp import prompt
from models import FinancialScore
from functions import save_score_to_csv, read_csv_from_s3, obtener_margen, ajuste_por_volumen
import os
import shutil
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="RAG Agent for financial Analysis", description="API para RAG Copilot", version="1.0.0")

origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # permite frontend
    allow_credentials=True,
    allow_methods=["*"],     # permite todos los métodos, incluido OPTIONS
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
PERSIST_DIR = "./storage"
groq_api_key = os.getenv("GROQ_API_KEY")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# Configurar modelo de embedding y LLM
embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-base")
llm = Groq(model="llama3-8b-8192", temperature=0,api_key=groq_api_key)

# Modelo LLM local usando llama-cpp y archivo GGUF (ajustar ruta a tu modelo local)
# llm = OpenAI(
#     model="o3-mini",  # o el modelo que uses
#     api_key=os.getenv("GROQ_API_KEY"),
#     base_url="https://api.groq.com/openai/v1"
# )
# llm = Ollama(model="llama3.2:latest")

# Aplicar configuración global moderna
Settings.embed_model = embed_model
Settings.llm = llm

collection_name = "copptech_documents"
# === Inicializar Qdrant o fallback local ===
use_qdrant = True
try:
    qdrant_client = QdrantClient(host="localhost", port=6333)
    # test conexión
    qdrant_client.get_collections()

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=768, 
                distance=Distance.COSINE
            )
        )
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
except Exception as e:
    print(f"[INFO] Qdrant no disponible. Usando almacenamiento local: {str(e)}")
    vector_store = None
    use_qdrant = False

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    '''
    Carga archivo pdf, este es almacenado en local y vectorizado en Qdrant
    '''
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Guardar el archivo localmente
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Leer el documento
        documents = SimpleDirectoryReader(input_dir=UPLOAD_DIR).load_data()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # Indexar con LlamaIndex + Qdrant
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            storage_context=storage_context
        )

        index.storage_context.persist(persist_dir=PERSIST_DIR)

        return JSONResponse(content={"status": "success", "message": f"Documento '{file.filename}' cargado y vectorizado"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el documento: {str(e)}")


@app.get("/query")
def query_documents(q: str = Query(..., description="Pregunta sobre los documentos")):
    '''
    Consulta a LLM RAG, pregunta sobre el contexto dado
    '''
    try:
        # Validar existencia del índice persistido
        if not use_qdrant:
            docstore_path = os.path.join(PERSIST_DIR, "docstore.json")
            if not os.path.exists(docstore_path):
                raise HTTPException(
                    status_code=404,
                    detail="No hay documentos indexados. Carga primero algunos PDFs."
                )

        # Cargar índice (ya sea local o Qdrant)
        storage_context = StorageContext.from_defaults(
            persist_dir=PERSIST_DIR if not use_qdrant else None,
            vector_store=vector_store if use_qdrant else None
        )
        index = load_index_from_storage(storage_context)

        # Ejecutar consulta
        query_engine = index.as_query_engine()
        response = query_engine.query(q)

        return {"query": q, "response": str(response)}

    except HTTPException as he:
        raise he  # Re-emitir errores personalizados
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la consulta: {str(e)}")


    
@app.get("/score_cliente", response_model=FinancialScore)
def score_cliente():
    '''Dado el contexto (Informes financieros) LLM genera un scoring financiero'''
    try:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR, vector_store=vector_store)
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query(prompt)
        parsed = FinancialScore.model_validate_json(str(response))
        save_score_to_csv(parsed)
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar score financiero: {str(e)}")

@app.get("/list-documents/")
async def list_documents():
    """
    Devuelve la lista de archivos PDF cargados y estado del índice vectorial.
    """
    try:
        # 1. Archivos PDF subidos
        uploaded_files = [
            f for f in os.listdir(UPLOAD_DIR)
            if f.endswith(".pdf")
        ]

        # 2. Estado del índice vectorial
        if use_qdrant:
            # Si estás usando Qdrant, contamos los vectores
            collection_info = qdrant_client.get_collection(collection_name)
            num_vectors = collection_info.vectors_count
            index_info = {
                "backend": "qdrant",
                "collection_name": collection_name,
                "num_vectors": num_vectors
            }
        else:
            # Modo local: contar cuántos archivos están persistidos
            persisted = os.listdir(PERSIST_DIR) if os.path.exists(PERSIST_DIR) else []
            index_info = {
                "backend": "local",
                "persisted_files": persisted,
                "num_items": len(persisted)
            }

        return {
            "uploaded_files": uploaded_files,
            "index_info": index_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar documentos: {str(e)}")

@app.post("/reset-index/")
async def reset_index():
    """
    Elimina todos los archivos subidos y reinicia el índice.
    Si se usa Qdrant, también elimina y recrea la colección.
    """
    try:
        # Borrar todos los archivos de uploads/
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Borrar almacenamiento local si existe
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        os.makedirs(PERSIST_DIR, exist_ok=True)

        # Si se usa Qdrant, eliminar y recrear la colección
        if use_qdrant and qdrant_client.collection_exists(collection_name):
            qdrant_client.delete_collection(collection_name=collection_name)
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

        return JSONResponse(content={"status": "success", "message": "Índice reiniciado correctamente."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al reiniciar el índice: {str(e)}")
# @app.get("/pricing_inform")
# def calcular_pricing(commodity: str = Query(..., description="Nombre del commodity (e.g., Copper)")):
#     '''
#     Informe basado en el score financiero dado por el LLM, la prediccion de ventas y el precio del cobre (archivos deben existir en bucket S3)
#     '''
#     try:
#         df_score = read_csv_from_s3("scores.csv")
#         df_demanda = read_csv_from_s3("prediction_sales.csv")
#         df_commodity = read_csv_from_s3("Copper_sim.csv") 

#         precio_p90 = df_commodity[df_commodity["Commodity"] == commodity]["upper"].values[0]
#         precio_p50 = df_commodity[df_commodity["Commodity"] == commodity]["mean"].values[0]
#         precio_p10 = df_commodity[df_commodity["Commodity"] == commodity]["lower"].values[0]
#         consumo_kg_por_litro = 0.1
#         costo_unitario_up = precio_p90 * consumo_kg_por_litro
#         costo_unitario_down = precio_p50 * consumo_kg_por_litro
#         costo_unitario_mean =precio_p10 * consumo_kg_por_litro

#         score =float(df_score.set_index("indicador").loc['score'].values[0])
#         margen = obtener_margen(score)
#         demanda_cmpc = df_demanda[df_demanda["cliente"] == "cmpc"]["pred_unidades"].sum()
#         ajuste_volumen = ajuste_por_volumen(demanda_cmpc)

#         precio_unitario_up = costo_unitario_up * (1 + margen + ajuste_volumen)
#         precio_unitario_down = costo_unitario_down * (1 + margen + ajuste_volumen)
#         precio_unitario_mean = costo_unitario_mean * (1 + margen + ajuste_volumen)

#         informe = DataFrame({
#             "cliente": ["cmpc"],
#             "score": [score],
#             "demanda_total_litros": [demanda_cmpc],
#             "costo_unitario_mean": [costo_unitario_mean],
#             "margen": [margen],
#             "ajuste_volumen": [ajuste_volumen],
#             "precio_unitario_up": [precio_unitario_up],
#             "precio_unitario_down": [precio_unitario_down],
#             "precio_unitario_mean": [precio_unitario_mean]
#         })

#         informe.to_csv("informe_pricing.csv")
#         return JSONResponse(status_code=200,content="Informe Generado")
    
#     except Exception as e:
#         return {"error": str(e)}
    
@app.get("/qdrant-status")
def qdrant_status():
    '''
    obtiene informacion de status qdrant
    '''
    try:
        collections = qdrant_client.get_collections()
        count = qdrant_client.count(collection_name=collection_name, exact=True).count
        return {"collections": [c.name for c in collections.collections], "count": count}
    except Exception as e:
        return {"error": str(e)}
    

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)