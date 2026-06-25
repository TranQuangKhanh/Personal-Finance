import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List

# =====================================================
# CONFIG
# =====================================================

KNOWLEDGE_BASE_PATH = "knowledge_base/finance_knowledge.txt"
CHROMA_DB_PATH = "knowledge_base/chroma_db"
COLLECTION_NAME = "finance_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"  # lightweight, fast, good quality
CHUNK_SIZE = 300   # characters per chunk
CHUNK_OVERLAP = 50 # overlap between chunks

# =====================================================
# STEP 1: LOAD & CHUNK DOCUMENT
# =====================================================

def load_and_chunk(path: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by double newline first (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        # Skip section headers
        if para.startswith("=") or para.startswith("#"):
            continue
        # If paragraph is short enough, keep as is
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # Sliding window chunking
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start += chunk_size - overlap

    print(f">>> Loaded {len(chunks)} chunks from {path}")
    return chunks

# =====================================================
# STEP 2: BUILD VECTOR STORE
# =====================================================

def build_vector_store(chunks: List[str], reset: bool = False):
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Reset collection if requested
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(">>> Existing collection deleted.")
        except:
            pass

    # Check if collection already exists and has data
    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0 and not reset:
            print(f">>> Collection '{COLLECTION_NAME}' already exists with {collection.count()} chunks. Skipping rebuild.")
            return collection
    except:
        pass

    # Build embeddings
    print(f">>> Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(">>> Embedding chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()

    # Create collection and add documents
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    print(f">>> Vector store built with {collection.count()} chunks.")
    return collection

# =====================================================
# STEP 3: RETRIEVE RELEVANT CHUNKS
# =====================================================

def retrieve(query: str, n_results: int = 3) -> List[str]:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    model = SentenceTransformer(EMBED_MODEL)
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    chunks = results["documents"][0]
    return chunks

# =====================================================
# STEP 4: FORMAT CONTEXT FOR LLM
# =====================================================

def get_context(query: str, n_results: int = 3) -> str:
    chunks = retrieve(query, n_results=n_results)
    context = "\n\n---\n\n".join(chunks)
    return context

# =====================================================
# MAIN — build vector store on first run
# =====================================================

if __name__ == "__main__":
    chunks = load_and_chunk(KNOWLEDGE_BASE_PATH)
    collection = build_vector_store(chunks, reset=True)

    # Quick test
    print("\n>>> Test retrieval:")
    test_query = "I have high debt and negative cash flow, what should I do?"
    results = retrieve(test_query, n_results=3)
    for i, chunk in enumerate(results):
        print(f"\n[Chunk {i+1}]:\n{chunk}")