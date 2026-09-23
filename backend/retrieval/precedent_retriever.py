"""Precedent retriever using OpenAI Embeddings + ChromaDB + Multi-Query + RRF."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = PROJECT_ROOT / "data" / "cases"
PERSIST_DIR = PROJECT_ROOT / "vector_db" / "chroma_cases"

_db = None


# ──────────────────────────────────────────────────────────────────
# DATABASE SETUP
# ──────────────────────────────────────────────────────────────────

def _get_embedding_model() -> OpenAIEmbeddings:
	return OpenAIEmbeddings(model="text-embedding-3-small")


def _get_db() -> Chroma:
	"""Load or create the ChromaDB vector store for case precedents."""
	global _db
	if _db is not None:
		return _db

	persist_path = str(PERSIST_DIR)
	embedding_model = _get_embedding_model()

	# Check if the database already exists on disk
	if PERSIST_DIR.exists():
		print("[precedent_retriever] Loading existing ChromaDB from disk...")
		_db = Chroma(
			persist_directory=persist_path,
			embedding_function=embedding_model,
			collection_name="cases",
		)
		doc_count = _db._collection.count()
		print(f"[precedent_retriever] Loaded {doc_count} chunks from ChromaDB.")
		return _db

	# Build from scratch
	print("[precedent_retriever] ChromaDB not found. Building from scratch...")
	_db = _build_vector_store()
	return _db


def _build_vector_store() -> Chroma:
	"""Load case .txt files, chunk them, embed, and store in ChromaDB."""
	if not CASES_DIR.exists():
		raise FileNotFoundError(f"Cases directory not found: {CASES_DIR}")

	# Step 1: Load all .txt files
	print(f"[precedent_retriever] Loading case files from {CASES_DIR}...")
	loader = DirectoryLoader(
		str(CASES_DIR),
		glob="*.txt",
		loader_cls=TextLoader,
		loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
		show_progress=True,
	)
	documents = loader.load()
	print(f"[precedent_retriever] Loaded {len(documents)} case files.")

	# Step 2: Chunk the documents
	print("[precedent_retriever] Splitting documents into chunks...")
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=800,
		chunk_overlap=100,
		separators=["\n\n", "\n", ". ", " ", ""],
	)
	chunks = text_splitter.split_documents(documents)
	print(f"[precedent_retriever] Created {len(chunks)} chunks.")

	# Step 3: Embed and store in ChromaDB
	print("[precedent_retriever] Embedding chunks and storing in ChromaDB...")
	print("[precedent_retriever] This will take ~15 minutes for 73K chunks. Be patient!")

	embedding_model = _get_embedding_model()
	db = Chroma.from_documents(
		documents=chunks,
		embedding=embedding_model,
		persist_directory=str(PERSIST_DIR),
		collection_name="cases",
	)
	print(f"[precedent_retriever] ✅ ChromaDB built with {db._collection.count()} chunks.")
	return db


# ──────────────────────────────────────────────────────────────────
# INITIALIZATION
# ──────────────────────────────────────────────────────────────────

def initialize_precedent_retriever() -> Dict[str, object]:
	"""Initialize the precedent retriever. Builds the DB if it doesn't exist."""
	db = _get_db()
	chunk_count = db._collection.count()
	created = not PERSIST_DIR.exists() or chunk_count == 0
	return {
		"created": created,
		"case_count": chunk_count,
	}


# ──────────────────────────────────────────────────────────────────
# BASIC RETRIEVAL (MMR)
# ──────────────────────────────────────────────────────────────────

def retrieve_precedents(query: str, top_k: int = 15) -> List[Dict[str, object]]:
	"""Retrieve top-k precedent chunks using MMR for diversity."""
	if not query or not query.strip():
		raise ValueError("Query must be a non-empty string.")

	db = _get_db()

	print(f"[precedent_retriever] Retrieving top {top_k} precedents (MMR)...")
	retriever = db.as_retriever(
		search_type="mmr",
		search_kwargs={"k": top_k, "fetch_k": top_k * 3, "lambda_mult": 0.5},
	)

	docs = retriever.invoke(query)

	results = []
	for i, doc in enumerate(docs):
		source = doc.metadata.get("source", "unknown")
		case_id = Path(source).name if source != "unknown" else "unknown"
		results.append({
			"rank": i + 1,
			"case_id": case_id,
			"text": doc.page_content,
			"text_preview": doc.page_content[:300],
		})

	print(f"[precedent_retriever] Retrieved {len(results)} precedent chunks.")
	return results


# ──────────────────────────────────────────────────────────────────
# MULTI-QUERY RETRIEVAL + RECIPROCAL RANK FUSION
# ──────────────────────────────────────────────────────────────────

class QueryVariations(BaseModel):
	"""Pydantic model for structured query generation output."""
	queries: List[str]


def _generate_query_variations(query: str, n: int = 3) -> List[str]:
	"""Use the LLM to generate n variations of a legal query."""
	llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
	llm_structured = llm.with_structured_output(QueryVariations)

	prompt = (
		f"You are a legal search expert. Generate {n} different variations of the following "
		f"legal query. Each variation should approach the same legal issue from a different angle, "
		f"using different legal terminology or phrasing.\n\n"
		f"Original query:\n{query}\n\n"
		f"Return {n} alternative queries."
	)

	response = llm_structured.invoke(prompt)
	variations = response.queries[:n]

	print(f"[precedent_retriever] Generated {len(variations)} query variations:")
	for i, v in enumerate(variations, 1):
		print(f"  {i}. {v}")

	return variations


def _reciprocal_rank_fusion(chunk_lists: List[List], k: int = 60) -> List[tuple]:
	"""Apply RRF to merge multiple ranked lists into a single ranking."""
	rrf_scores = defaultdict(float)
	all_unique_chunks = {}

	for chunks in chunk_lists:
		for position, chunk in enumerate(chunks, 1):
			chunk_content = chunk.page_content
			all_unique_chunks[chunk_content] = chunk
			rrf_scores[chunk_content] += 1 / (k + position)

	sorted_chunks = sorted(
		[(all_unique_chunks[content], score) for content, score in rrf_scores.items()],
		key=lambda x: x[1],
		reverse=True,
	)

	return sorted_chunks


def retrieve_precedents_multi_query(query: str, top_k: int = 15) -> List[Dict[str, object]]:
	"""Multi-query retrieval with RRF fusion for better recall on complex legal queries."""
	if not query or not query.strip():
		raise ValueError("Query must be a non-empty string.")

	db = _get_db()

	# Step 1: Generate query variations
	print("[precedent_retriever] Running multi-query retrieval...")
	variations = _generate_query_variations(query)

	# Step 2: Retrieve for each variation using MMR
	retriever = db.as_retriever(
		search_type="mmr",
		search_kwargs={"k": top_k, "fetch_k": top_k * 3, "lambda_mult": 0.5},
	)

	all_results = []
	for i, variation in enumerate(variations, 1):
		docs = retriever.invoke(variation)
		all_results.append(docs)
		print(f"[precedent_retriever] Query {i} returned {len(docs)} chunks.")

	# Step 3: Fuse with RRF
	fused = _reciprocal_rank_fusion(all_results, k=60)
	print(f"[precedent_retriever] RRF fused {len(fused)} unique chunks.")

	# Step 4: Format top results
	results = []
	for i, (doc, rrf_score) in enumerate(fused[:top_k]):
		source = doc.metadata.get("source", "unknown")
		case_id = Path(source).name if source != "unknown" else "unknown"
		results.append({
			"rank": i + 1,
			"case_id": case_id,
			"rrf_score": round(rrf_score, 4),
			"text": doc.page_content,
			"text_preview": doc.page_content[:300],
		})

	print(f"[precedent_retriever] ✅ Multi-query retrieval returned {len(results)} precedents.")
	return results


if __name__ == "__main__":
	_build_vector_store()
