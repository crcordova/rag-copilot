# AI Agent de Análisis Financiero con RAG
Esta es una API desarrollada con FastAPI que permite cargar informes financieros en PDF, vectorizarlos usando LlamaIndex y Qdrant, y luego consultar la información mediante un modelo LLM para generar scoring financiero y análisis de pricing dinámico, apoyado por simulaciones de precios de commodities y predicciones de ventas.

## Tecnologías utilizadas

 - FastAPI – Framework backend moderno y asincrónico

 - LlamaIndex – Framework RAG (Retrieval-Augmented Generation)

 - Qdrant – Base de datos vectorial para indexación semántica

 - Groq (LLaMA3 8B) – Modelo LLM para razonamiento financiero

 - PDF ingestion – Carga y vectorización automática de documentos

 - Amazon S3 – Fuente de datos para simulaciones y ventas

 ## Funciones Principales

 - `/upload-pdf/` : Carga un archivo PDF de informe financiero, lo guarda localmente y lo vectoriza en Qdrant para consultas posteriores.
 - ` /query?q=...`: Consulta abierta a los documentos cargados mediante LLM + RAG, útil para preguntar sobre contenido específico de los informes.
 - `/score_cliente`: Genera automáticamente un score financiero en base al contenido de los informes vectorizados. El resultado se valida con un modelo Pydantic y se guarda en CSV.
 - `/pricing_inform?commodity=Coppe`:Calcula un informe de pricing utilizando:

    - Predicción de demanda
    - Simulación de precios de commodities
    - Score financiero

## Configuración inicial
### Clonar repositorio

```bash
 git clone https://github.com/crcordova/rag-copilot
 cd rag-copilot
```
### Instalar dependencias
```bash
 pip install -r requirements.txt
```

### Configuración variables de entonro
```bash
cp .env.example .env
```

## Configuración Qdrant

### Ejecutar Qdran con Docker
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Flujo General
 1. Subes un archivo PDF con `/upload-pdf/`
 2. Se vectoriza el documento y se guarda en Qdrant
 3. Puedes:
    - Hacer preguntas con `/query?q=...`

    - Generar score financiero con `/score_cliente`

    - Obtener informe de pricing con `/pricing_inform?commodity=Copper`

## Datos utilizados (S3)
El endpoint de pricing depende de archivos almacenados en un bucket de S3:

- scores.csv → Resultado del score financiero

- prediction_sales.csv → Predicción de demanda por cliente

- Copper_sim.csv → Simulación de precios del commodity (P10, P50, P90)

Asegúrate de que esos archivos existan en tu bucket configurado con las credenciales AWS del `.env`.

## Requisitos
 - Python 3.10+

 - Qdrant (local o remoto)

 - Cuenta Groq (https://console.groq.com/)

 - Bucket de S3 configurado con los archivos necesarios