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

        # Indexar con LlamaIndex + Qdrant
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store
        )

        index.storage_context.persist(persist_dir=PERSIST_DIR)

        return JSONResponse(content={"status": "success", "message": f"Documento '{file.filename}' cargado y vectorizado"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el documento: {str(e)}")

@app.get("/query")
def query_documents(q: str = Query(..., description="Pregunta sobre los documentos")):
    try:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, vector_store=vector_store)
        query_engine = index.as_query_engine()
        response = query_engine.query(q)
        return {"query": q, "response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la consulta: {str(e)}")
