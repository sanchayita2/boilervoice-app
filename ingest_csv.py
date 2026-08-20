import pandas as pd
import chromadb

def ingest_csv_to_chroma(csv_path="boiler_inspection_data.csv"):
    df = pd.read_csv(csv_path)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="boiler_telemetry")

    documents, metadatas, ids = [], [], []
    for idx, row in df.iterrows():
        # Represent each row as descriptive text for vector similarity search
        doc_text = f"Boiler {row['Boiler_ID']} at {row['Location']} status is {row['Status']}. Temp: {row['Temperature_C']}C, Pressure: {row['Pressure_PSI']} PSI."
        documents.append(doc_text)
        metadatas.append({"boiler_id": str(row['Boiler_ID'])})
        ids.append(f"row_{idx}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print("✅ CSV ingested into ChromaDB 'boiler_telemetry' collection.")

if __name__ == "__main__":
    ingest_csv_to_chroma()