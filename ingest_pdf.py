import os
import chromadb
from dotenv import find_dotenv, load_dotenv
from pypdf import PdfReader

load_dotenv(find_dotenv())


def ingest_pdf_manual(
    pdf_path="boiler_manual.pdf", collection_name="boiler_manuals"
):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file '{pdf_path}' not found.")

    print(f"📄 Extracting text from {pdf_path}...")
    reader = PdfReader(pdf_path)
    chunks = []

    # Read PDF page by page
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            chunks.append({"text": text.strip(), "page": page_num + 1})

    # Initialize persistent ChromaDB vector storage
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Insert text chunks into vector collection
    for idx, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk["text"]],
            metadatas=[{"page": chunk["page"], "source": pdf_path}],
            ids=[f"doc_{idx}"],
        )

    print(
        f"✅ Ingested {len(chunks)} pages/chunks into ChromaDB collection '{collection_name}'."
    )


if __name__ == "__main__":
    ingest_pdf_manual("boiler_manual.pdf")