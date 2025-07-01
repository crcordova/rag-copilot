from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.readers.wikipedia import WikipediaReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from pandas import DataFrame
from promp import prompt
from models import FinancialScore
from functions import save_score_to_csv, read_csv_from_s3, obtener_margen, ajuste_por_volumen
import os
import shutil

# Inicializar FastAPI
app = FastAPI()

# Configuración de directorios
UPLOAD_DIR = "./uploads"
PERSIST_DIR = "./storage"
groq_api_key = os.getenv("GROQ_API_KEY")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)


# Configurar Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "copptech_documents"

if not qdrant_client.collection_exists(collection_name):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=768,  # tamaño del embedding
            distance=Distance.COSINE
        )
    )

embed_model = HuggingFaceEmbedding(model_name="hkunlp/instructor-large")

# Modelo LLM local usando llama-cpp y archivo GGUF (ajustar ruta a tu modelo local)
# llm = OpenAI(
#     model="o3-mini",  # o el modelo que uses
#     api_key=os.getenv("GROQ_API_KEY"),
#     base_url="https://api.groq.com/openai/v1"
# )
llm = Groq(model="llama3-8b-8192", temperature=0,api_key=groq_api_key)
# llm = Ollama(model="llama3.2:latest")

# Aplicar configuración global moderna
Settings.embed_model = embed_model
Settings.llm = llm

vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
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

        # index = VectorStoreIndex.from_documents(
        #     documents,
        #     storage_context=storage_context
        # )

        index.storage_context.persist(persist_dir=PERSIST_DIR)

        return JSONResponse(content={"status": "success", "message": f"Documento '{file.filename}' cargado y vectorizado"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el documento: {str(e)}")

@app.get("/query")
def query_documents(q: str = Query(..., description="Pregunta sobre los documentos")):
    try:
        # storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        # index = load_index_from_storage(storage_context, vector_store=vector_store)
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR, vector_store=vector_store)
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine()
        response = query_engine.query(q)
        return {"query": q, "response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la consulta: {str(e)}")
    
@app.get("/score_cliente", response_model=FinancialScore)
def score_cliente():
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


@app.get("/pricing_inform")
def calcular_pricing(commodity: str = Query(..., description="Nombre del commodity (e.g., Copper)")):
    try:
        df_score = read_csv_from_s3("scores.csv")
        df_demanda = read_csv_from_s3("prediction_sales.csv")
        df_commodity = read_csv_from_s3("Copper_sim.csv") 

        precio_p90 = df_commodity[df_commodity["Commodity"] == commodity]["upper"].values[0]
        precio_p50 = df_commodity[df_commodity["Commodity"] == commodity]["mean"].values[0]
        precio_p10 = df_commodity[df_commodity["Commodity"] == commodity]["lower"].values[0]
        consumo_kg_por_litro = 0.1
        costo_unitario_up = precio_p90 * consumo_kg_por_litro
        costo_unitario_down = precio_p50 * consumo_kg_por_litro
        costo_unitario_mean =precio_p10 * consumo_kg_por_litro

        score =float(df_score.set_index("indicador").loc['score'].values[0])
        margen = obtener_margen(score)
        demanda_cmpc = df_demanda[df_demanda["cliente"] == "cmpc"]["pred_unidades"].sum()
        ajuste_volumen = ajuste_por_volumen(demanda_cmpc)

        precio_unitario_up = costo_unitario_up * (1 + margen + ajuste_volumen)
        precio_unitario_down = costo_unitario_down * (1 + margen + ajuste_volumen)
        precio_unitario_mean = costo_unitario_mean * (1 + margen + ajuste_volumen)

        informe = DataFrame({
            "cliente": ["cmpc"],
            "score": [score],
            "demanda_total_litros": [demanda_cmpc],
            "costo_unitario_mean": [costo_unitario_mean],
            "margen": [margen],
            "ajuste_volumen": [ajuste_volumen],
            "precio_unitario_up": [precio_unitario_up],
            "precio_unitario_down": [precio_unitario_down],
            "precio_unitario_mean": [precio_unitario_mean]
        })

        informe.to_csv("informe_pricing.csv")
        return JSONResponse(status_code=200,content="Informe Generado")
    
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/qdrant-status")
def qdrant_status():
    try:
        collections = qdrant_client.get_collections()
        count = qdrant_client.count(collection_name=collection_name, exact=True).count
        return {"collections": [c.name for c in collections.collections], "count": count}
    except Exception as e:
        return {"error": str(e)}