import os
import pandas as pd
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

google_ef = embedding_functions.GoogleGeminiEmbeddingFunction(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name="models/text-embedding-004"
)

def ingest_csv_to_chroma(csv_path="boiler_inspection_data.csv"):
    df = pd.read_csv(csv_path)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="boiler_telemetry", 
        embedding_function=google_ef
    )

    documents, metadatas, ids = [], [], []
    for idx, row in df.iterrows():
        doc_text = f"Boiler {row['Boiler_ID']} at {row['Location']} status is {row['Status']}. Temp: {row['Temperature_C']}C, Pressure: {row['Pressure_PSI']} PSI."
        documents.append(doc_text)
        metadatas.append({"boiler_id": str(row['Boiler_ID'])})
        ids.append(f"row_{idx}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print("✅ CSV ingested into ChromaDB 'boiler_telemetry' collection with Gemini embeddings.")