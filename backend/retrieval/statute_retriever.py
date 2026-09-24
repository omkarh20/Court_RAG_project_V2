"""Statute retriever using OpenAI Embeddings + ChromaDB + Hybrid Search."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATUTES_DIR = PROJECT_ROOT / "data" / "statutes"
PERSIST_DIR = PROJECT_ROOT / "vector_db" / "chroma_statutes"

_db = None
_bm25_retriever = None


# ──────────────────────────────────────────────────────────────────
# DATABASE SETUP
# ──────────────────────────────────────────────────────────────────

def _get_embedding_model() -> OpenAIEmbeddings:
	return OpenAIEmbeddings(model="text-embedding-3-small")


def _get_db() -> Chroma:
	"""Load or create the ChromaDB vector store for statutes."""
	global _db
	if _db is not None:
		return _db

	persist_path = str(PERSIST_DIR)
	embedding_model = _get_embedding_model()

	# Check if the database already exists on disk
	if PERSIST_DIR.exists():
		print("[statute_retriever] Loading existing ChromaDB from disk...")
		_db = Chroma(
			persist_directory=persist_path,
			embedding_function=embedding_model,
			collection_name="statutes",
		)
		doc_count = _db._collection.count()
		print(f"[statute_retriever] Loaded {doc_count} chunks from ChromaDB.")
		return _db

	# Build from scratch
	print("[statute_retriever] ChromaDB not found. Building from scratch...")
	_db = _build_vector_store()
	return _db


def _build_vector_store() -> Chroma:
	"""Load statute .txt files, chunk them, embed, and store in ChromaDB."""
	if not STATUTES_DIR.exists():
		raise FileNotFoundError(f"Statutes directory not found: {STATUTES_DIR}")

	# Step 1: Load all .txt files
	print(f"[statute_retriever] Loading statute files from {STATUTES_DIR}...")
	loader = DirectoryLoader(
		str(STATUTES_DIR),
		glob="*.txt",
		loader_cls=TextLoader,
		loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
		show_progress=True,
	)
	documents = loader.load()
	print(f"[statute_retriever] Loaded {len(documents)} statute files.")

	# Step 2: Chunk the documents (smaller chunks for statutes)
	print("[statute_retriever] Splitting documents into chunks...")
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=500,
		chunk_overlap=50,
		separators=["\n\n", "\n", ". ", " ", ""],
	)
	chunks = text_splitter.split_documents(documents)
	print(f"[statute_retriever] Created {len(chunks)} chunks.")

	# Step 3: Embed and store in ChromaDB
	print("[statute_retriever] Embedding chunks and storing in ChromaDB...")

	embedding_model = _get_embedding_model()
	db = Chroma.from_documents(
		documents=chunks,
		embedding=embedding_model,
		persist_directory=str(PERSIST_DIR),
		collection_name="statutes",
	)
	print(f"[statute_retriever] ✅ ChromaDB built with {db._collection.count()} chunks.")
	return db


# ──────────────────────────────────────────────────────────────────
# BM25 INDEX
# ──────────────────────────────────────────────────────────────────

def _build_bm25_retriever(top_k: int = 5) -> BM25Retriever:
	"""Build a BM25 retriever from all statute chunks. Cached after first call."""
	global _bm25_retriever

	if _bm25_retriever is not None:
		_bm25_retriever.k = top_k
		return _bm25_retriever

	db = _get_db()
	print("[statute_retriever] Building BM25 index from ChromaDB chunks...")

	all_data = db.get(include=["documents", "metadatas"])
	docs = [
		Document(page_content=doc, metadata=meta)
		for doc, meta in zip(all_data["documents"], all_data["metadatas"])
	]

	print(f"[statute_retriever] Building BM25 index over {len(docs)} chunks...")
	_bm25_retriever = BM25Retriever.from_documents(docs)
	_bm25_retriever.k = top_k
	print("[statute_retriever] ✅ BM25 index built.")
	return _bm25_retriever


# ──────────────────────────────────────────────────────────────────
# HYBRID RETRIEVER
# ──────────────────────────────────────────────────────────────────

def _get_hybrid_retriever(top_k: int = 5) -> EnsembleRetriever:
	"""Create a hybrid retriever combining semantic search with BM25."""
	db = _get_db()

	semantic = db.as_retriever(
		search_type="mmr",
		search_kwargs={"k": top_k, "fetch_k": top_k * 3, "lambda_mult": 0.5},
	)

	bm25 = _build_bm25_retriever(top_k=top_k)

	hybrid = EnsembleRetriever(
		retrievers=[semantic, bm25],
		weights=[0.5, 0.5],
	)
	return hybrid


# ──────────────────────────────────────────────────────────────────
# INITIALIZATION
# ──────────────────────────────────────────────────────────────────

def initialize_statute_retriever() -> Dict[str, object]:
	"""Initialize the statute retriever. Builds the DB and BM25 index if needed."""
	db = _get_db()
	chunk_count = db._collection.count()

	# Pre-build the BM25 index at startup
	_build_bm25_retriever()

	created = not PERSIST_DIR.exists() or chunk_count == 0
	return {
		"created": created,
		"statute_count": chunk_count,
	}


# ──────────────────────────────────────────────────────────────────
# RETRIEVAL (HYBRID)
# ──────────────────────────────────────────────────────────────────

def retrieve_statutes(query: str, top_k: int = 5) -> List[Dict[str, object]]:
	"""Retrieve top-k statute chunks using hybrid search (Semantic + BM25)."""
	if not query or not query.strip():
		raise ValueError("Query must be a non-empty string.")

	print(f"[statute_retriever] Retrieving top {top_k} statutes (Hybrid)...")
	hybrid = _get_hybrid_retriever(top_k=top_k)
	docs = hybrid.invoke(query)

	results = []
	for i, doc in enumerate(docs[:top_k]):
		source = doc.metadata.get("source", "unknown")
		filename = Path(source).name if source != "unknown" else "unknown"
		results.append({
			"rank": i + 1,
			"filename": filename,
			"title": filename.replace(".txt", ""),
			"text": doc.page_content,
			"description": doc.page_content[:300],
		})

	print(f"[statute_retriever] Retrieved {len(results)} statute chunks.")
	return results


if __name__ == "__main__":
	_build_vector_store()
