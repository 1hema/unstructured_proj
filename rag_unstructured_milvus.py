"""
RAG application using unstructured.io to parse PDFs and Milvus (pymilvus) for vector storage.

Features:
- Parse PDF files with unstructured.partition_pdf
- Chunk text into overlapping chunks
- Compute embeddings using sentence-transformers by default (configurable)
- Store vectors + metadata in Milvus
- Query Milvus for nearest neighbors and return matching text chunks

Files included in this single script:
- A full Python implementation (functions + CLI)
- A small embedded README in the header explaining usage

Usage examples (after installing requirements):

# Ingest PDFs from ./pdfs into Milvus
python rag_unstructured_milvus.py ingest --pdf-dir ./pdfs --collection my_rag_collection

# Query
python rag_unstructured_milvus.py query --collection my_rag_collection --q "What is the refund policy?"

Environment variables (optional):
- MILVUS_HOST (default: localhost)
- MILVUS_PORT (default: 19530)
- EMBEDDING_MODEL (default: sentence-transformers/all-MiniLM-L6-v2)

Requirements:
- unstructured (for PDF parsing)
- pymilvus
- sentence-transformers
- tqdm

Note: If you prefer OpenAI embeddings, you can modify get_embeddings() to call OpenAI's embeddings instead (the code includes a commented stub on how to swap).

"""

import os
import argparse
import glob
import math
from typing import List, Dict, Tuple
from tqdm import tqdm

# PDF parsing
try:
    # unstructured: partition_pdf or partition
    from unstructured.partition.pdf import partition_pdf
except Exception:
    # in some versions: from unstructured.partition import partition
    try:
        from unstructured.partition import partition
        partition_pdf = lambda filename: partition(filename)
    except Exception:
        raise ImportError("unstructured library is required. Install with `pip install unstructured`")

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    raise ImportError("sentence-transformers is required. Install with `pip install sentence-transformers")

# Milvus client
try:
    from pymilvus import (
        connections,
        FieldSchema,
        CollectionSchema,
        DataType,
        Collection,
        utility,
        # for indexes
        Index,
    )
except Exception:
    raise ImportError("pymilvus is required. Install with `pip install pymilvus`")

# --------------------------- Configuration ---------------------------
DEFAULT_MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
DEFAULT_MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
#DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","sentence-transformers/paraphrase-MiniLM-L3-v2")

DEFAULT_DIM = 384  # all-MiniLM-L6-v2 output dim

# --------------------------- Utilities ---------------------------

def connect_milvus(host: str = DEFAULT_MILVUS_HOST, port: str = DEFAULT_MILVUS_PORT):
    """Connect to Milvus and return connection info."""
    connections.connect(host=host, port=port)


def create_collection(collection_name: str, dim: int = DEFAULT_DIM, metric: str = "IP") -> Collection:
    """Create a Milvus collection (if not exist) with a vector field and metadata.

    Fields:
    - id (int64)
    - embedding (float_vector)
    - text (varchar)
    - source (varchar)
    """
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists - opening it.")
        return Collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
    ]
    schema = CollectionSchema(fields, description="RAG collection of PDF chunks")
    collection = Collection(name=collection_name, schema=schema)
    # Create an index on the embedding field
    index_params = {
        "index_type": "HNSW",
        "metric_type": metric,  # "IP" or "L2"
        "params": {"M": 8, "efConstruction": 64},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    return collection

# --------------------------- Text processing ---------------------------

def parse_pdf_to_text(pdf_path: str) -> str:
    """Use unstructured.partition_pdf to extract text from a PDF file and return as a single string."""
    
    #elements = partition_pdf(filename=pdf_path)

    elements = partition_pdf(
        filename=pdf_path,
        strategy="fast",        # lighter parsing, no full layout
        languages=["eng"],      # skip language auto-detection
        extract_images=False    # saves memory
    )


    # elements is a list of elements; extract text
    texts = []
    for el in elements:
        # Many element types have .text attribute
        txt = getattr(el, "text", None)
        if txt:
            texts.append(txt.strip())
    return "\n\n".join([t for t in texts if t])


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Simple whitespace-based chunking. chunk_size in words."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

# --------------------------- Embeddings ---------------------------

class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        # update dim automatically
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    # If you prefer OpenAI, replace the implementation of embed_batch with an OpenAI call.

# --------------------------- Milvus operations ---------------------------

def upsert_chunks_to_milvus(collection: Collection, chunks: List[Tuple[str, str]], embedder: Embedder, batch_size: int = 64):
    """chunks: list of tuples (text, source)
    Inserts into Milvus with auto-generated ids.
    """
    texts = [c[0] for c in chunks]
    sources = [c[1] for c in chunks]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_sources = sources[i : i + batch_size]
        embeddings = embedder.embed_batch(batch_texts)
        # Prepare data for Milvus: fields ordering matches schema
        entities = [embeddings, batch_texts, batch_sources]
        collection.insert(entities)
    collection.flush()


def search(collection: Collection, query: str, embedder: Embedder, top_k: int = 5) -> List[Dict]:
    q_emb = embedder.embed_batch([query])[0]
    # build search params; ef is the search param for HNSW
    search_params = {"metric_type": "IP", "params": {"ef": 64}}
    res = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=None,
        output_fields=["text", "source"],
    )
    hits = []
    for hits_per_query in res:
        for hit in hits_per_query:
            hits.append({
                "id": hit.id,
                "score": float(hit.distance),
                "text": hit.entity.get("text"),
                "source": hit.entity.get("source"),
            })
    return hits

# --------------------------- High-level functions ---------------------------

def ingest_pdfs(pdf_dir: str, collection_name: str, chunk_size: int = 500, chunk_overlap: int = 50, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    connect_milvus()
    # load embedder first
    embedder = Embedder(embedding_model)
    collection = create_collection(collection_name, dim=embedder.dim)

    pdf_files = glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True)
    if not pdf_files:
        print("No PDF files found in the provided directory.")
        return

    all_chunks = []
    for pdf in tqdm(pdf_files, desc="Parsing PDFs"):
        try:
            text = parse_pdf_to_text(pdf)
        except Exception as e:
            print(f"Failed to parse {pdf}: {e}")
            continue
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # tag each chunk with source (filename)
        chunks_with_source = [(c, os.path.basename(pdf)) for c in chunks]
        all_chunks.extend(chunks_with_source)

    print(f"Total chunks to upsert: {len(all_chunks)}")
    upsert_chunks_to_milvus(collection, all_chunks, embedder)
    print("Ingestion complete.")


def query_collection(collection_name: str, query: str, top_k: int = 5, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    connect_milvus()
    embedder = Embedder(embedding_model)
    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist.")
    collection = Collection(collection_name)
    collection.load()
    results = search(collection, query, embedder, top_k=top_k)
    return results

# --------------------------- CLI ---------------------------

def make_arg_parser():
    p = argparse.ArgumentParser(description="RAG with unstructured + Milvus")
    sub = p.add_subparsers(dest="cmd")

    ingest = sub.add_parser("ingest", help="Ingest PDFs into Milvus")
    ingest.add_argument("--pdf-dir")
    ingest.add_argument("--collection", required=True)
    ingest.add_argument("--chunk-size", type=int, default=500)
    ingest.add_argument("--chunk-overlap", type=int, default=50)
    ingest.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)

    query = sub.add_parser("query", help="Query a collection")
    query.add_argument("--collection", required=True)
    query.add_argument("--q", required=True)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)

    return p


def main():
    #parser = make_arg_parser()
    #args = parser.parse_args()

    pdf_dir = "/Users/hema/Desktop/Prototype_1012/raw_data/"
    collection = "my_rag_collection"
    chunk_size = 500
    chunk_overlap = 50
    EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL

    ingest_pdfs(pdf_dir, collection, chunk_size, chunk_overlap, EMBEDDING_MODEL)

    """ if args.cmd == "ingest":
        # ingest_pdfs(args.pdf_dir, args.collection, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, embedding_model=args.embedding_model)
        ingest_pdfs(pdf_dir, args.collection, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, embedding_model=args.embedding_model)

    elif args.cmd == "query":
        res = query_collection(args.collection, args.q, top_k=args.top_k, embedding_model=args.embedding_model)
        print("--- Top results ---")
        for r in res:
            print(f"score: {r['score']:.4f} | source: {r['source']}\n{r['text'][:400]}\n---\n")
    else:
        parser.print_help() """


if __name__ == "__main__":
    main()
