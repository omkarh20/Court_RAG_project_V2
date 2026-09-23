"""Statute retriever using OpenAI Embeddings + ChromaDB."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATUTES_DIR = PROJECT_ROOT / "data" / "statutes"
PERSIST_DIR = PROJECT_ROOT / "vector_db" / "chroma_statutes"

_db = None


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
# INITIALIZATION
# ──────────────────────────────────────────────────────────────────

def initialize_statute_retriever() -> Dict[str, object]:
	"""Initialize the statute retriever. Builds the DB if it doesn't exist."""
	db = _get_db()
	chunk_count = db._collection.count()
	created = not PERSIST_DIR.exists() or chunk_count == 0
	return {
		"created": created,
		"statute_count": chunk_count,
	}


# ──────────────────────────────────────────────────────────────────
# RETRIEVAL (MMR)
# ──────────────────────────────────────────────────────────────────

def retrieve_statutes(query: str, top_k: int = 5) -> List[Dict[str, object]]:
	"""Retrieve top-k statute chunks using MMR for diversity."""
	if not query or not query.strip():
		raise ValueError("Query must be a non-empty string.")

	db = _get_db()

	print(f"[statute_retriever] Retrieving top {top_k} statutes (MMR)...")
	retriever = db.as_retriever(
		search_type="mmr",
		search_kwargs={"k": top_k, "fetch_k": top_k * 3, "lambda_mult": 0.5},
	)

	docs = retriever.invoke(query)

	results = []
	for i, doc in enumerate(docs):
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
