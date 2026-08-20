import os
import sqlite3
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from google import genai

# Load environment variables
load_dotenv(find_dotenv())

# Initialize Gemini LLM Client
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Define Google Gemini API Embedding Function
google_ef = embedding_functions.GoogleGeminiEmbeddingFunction(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name="models/gemini-embedding-001"
)


def query_hybrid_rag(
    transcribed_text: str,
    db_path="boiler_data.db",
    collection_name="boiler_manuals",
) -> str:
    """Combines structured SQLite telemetry with unstructured ChromaDB vector manual context."""
    
    # 1. Fetch Structured Boiler Telemetry (SQLite)
    structured_context = ""
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM boiler_inspections", conn)
        conn.close()
        structured_context = df.to_csv(index=False)
    else:
        structured_context = "No SQLite telemetry database found."

    # 2. Fetch Unstructured Manual Excerpts (ChromaDB Vector Search via Gemini API)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name=collection_name, 
        embedding_function=google_ef
    )

    # Query vector store for the 2 most relevant manual pages
    results = collection.query(query_texts=[transcribed_text], n_results=2)
    retrieved_docs = results["documents"][0] if results["documents"] else []
    unstructured_context = "\n---\n".join(retrieved_docs)

    # 3. Build Unified Hybrid Prompt
    system_instruction = (
        "You are an industrial boiler inspection assistant. "
        "Use the structured boiler status to identify specific equipment states, and reference the unstructured manual excerpts "
        "for troubleshooting procedures. Answer concisely in 2 natural sentences."
    )

    full_prompt = (
        f"STRUCTURED TELEMETRY DATA:\n{structured_context}\n\n"
        f"UNSTRUCTURED MANUAL EXCERPTS:\n{unstructured_context}\n\n"
        f"USER QUESTION: {transcribed_text}"
    )

    print("🤖 Querying Gemini with Hybrid Context...")

    response = genai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=full_prompt,
        config={"system_instruction": system_instruction},
    )

    answer = response.text.strip()
    print(f"💡 Gemini Answer: '{answer}'")
    return answer


if __name__ == "__main__":
    query_hybrid_rag(
        "Boiler 3 has high temperature. What procedure should I follow?"
    )