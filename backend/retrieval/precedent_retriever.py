"""Precedent retriever using OpenAI Embeddings + ChromaDB + Hybrid Search + Role-Specific Retrieval."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = PROJECT_ROOT / "data" / "cases"
PERSIST_DIR = PROJECT_ROOT / "vector_db" / "chroma_cases"

_db = None
_bm25_retriever = None
_all_docs_cache = None


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
# BM25 INDEX (KEYWORD MATCHING)
# ──────────────────────────────────────────────────────────────────

def _build_bm25_retriever(top_k: int = 15) -> BM25Retriever:
	"""Build a BM25 retriever from all chunks in ChromaDB. Cached after first call."""
	global _bm25_retriever, _all_docs_cache

	if _bm25_retriever is not None:
		_bm25_retriever.k = top_k
		return _bm25_retriever

	db = _get_db()
	print("[precedent_retriever] Building BM25 index from ChromaDB chunks...")

	# Pull all documents from ChromaDB
	all_data = db.get(include=["documents", "metadatas"])
	documents = all_data["documents"]
	metadatas = all_data["metadatas"]

	# Convert to LangChain Document objects for BM25Retriever
	_all_docs_cache = [
		Document(page_content=doc, metadata=meta)
		for doc, meta in zip(documents, metadatas)
	]

	print(f"[precedent_retriever] Building BM25 index over {len(_all_docs_cache)} chunks...")
	_bm25_retriever = BM25Retriever.from_documents(_all_docs_cache)
	_bm25_retriever.k = top_k
	print("[precedent_retriever] ✅ BM25 index built.")
	return _bm25_retriever


# ──────────────────────────────────────────────────────────────────
# HYBRID RETRIEVER (SEMANTIC + BM25)
# ──────────────────────────────────────────────────────────────────

def _get_hybrid_retriever(top_k: int = 15) -> EnsembleRetriever:
	"""Create a hybrid retriever combining dense semantic search with sparse BM25."""
	db = _get_db()

	# Dense retriever (semantic + MMR for diversity)
	semantic = db.as_retriever(
		search_type="mmr",
		search_kwargs={"k": top_k, "fetch_k": top_k * 3, "lambda_mult": 0.5},
	)

	# Sparse retriever (BM25 keyword matching)
	bm25 = _build_bm25_retriever(top_k=top_k)

	# Combine with equal weights — EnsembleRetriever uses RRF internally
	hybrid = EnsembleRetriever(
		retrievers=[semantic, bm25],
		weights=[0.7, 0.3],
	)
	return hybrid


# ──────────────────────────────────────────────────────────────────
# INITIALIZATION
# ──────────────────────────────────────────────────────────────────

def initialize_precedent_retriever() -> Dict[str, object]:
	"""Initialize the precedent retriever. Builds the DB and BM25 index if needed."""
	db = _get_db()
	chunk_count = db._collection.count()

	# Pre-build the BM25 index at startup
	_build_bm25_retriever()

	created = not PERSIST_DIR.exists() or chunk_count == 0
	return {
		"created": created,
		"case_count": chunk_count,
	}


# ──────────────────────────────────────────────────────────────────
# BASIC RETRIEVAL (HYBRID)
# ──────────────────────────────────────────────────────────────────

def retrieve_precedents(query: str, top_k: int = 15) -> List[Dict[str, object]]:
	"""Retrieve top-k precedent chunks using hybrid search (Semantic + BM25)."""
	if not query or not query.strip():
		raise ValueError("Query must be a non-empty string.")

	print(f"[precedent_retriever] Retrieving top {top_k} precedents (Hybrid)...")
	hybrid = _get_hybrid_retriever(top_k=top_k)
	docs = hybrid.invoke(query)

	results = _format_results(docs[:top_k])
	print(f"[precedent_retriever] Retrieved {len(results)} precedent chunks.")
	return results


# ──────────────────────────────────────────────────────────────────
# ROLE-SPECIFIC RETRIEVAL (INDEPENDENT AGENTIC SEARCH)
# ──────────────────────────────────────────────────────────────────

class QueryVariations(BaseModel):
	"""Pydantic model for structured query generation output."""
	queries: List[str]


_ROLE_QUERY_TEMPLATES = {
	"prosecution": (
		"You are a legal search expert working for the PROSECUTION.\n"
		"Generate {n} search queries to find precedent cases where the accused was CONVICTED "
		"for similar crimes. Focus on cases establishing intent, weapon use, and fatal outcomes.\n\n"
		"Facts of the current case:\n{facts}\n\n"
		"Return {n} search queries."
	),
	"defense": (
		"You are a legal search expert working for the DEFENSE.\n"
		"Generate {n} search queries to find precedent cases where the accused was ACQUITTED "
		"or received a REDUCED SENTENCE. Focus on provocation, lack of premeditation, "
		"self-defense, and mitigating circumstances.\n\n"
		"Facts of the current case:\n{facts}\n\n"
		"Return {n} search queries."
	),
}


def _generate_role_queries(facts: str, role: str, n: int = 3) -> List[str]:
	"""Generate role-specific query variations using the LLM."""
	llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
	llm_structured = llm.with_structured_output(QueryVariations)

	template = _ROLE_QUERY_TEMPLATES[role]
	prompt = template.format(n=n, facts=facts)

	response = llm_structured.invoke(prompt)
	variations = response.queries[:n]

	print(f"[precedent_retriever] Generated {len(variations)} {role} queries:")
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


def retrieve_for_role(facts: str, role: str, top_k: int = 10) -> List[Dict[str, object]]:
	"""Role-specific retrieval: prosecution and defense each get their own tailored precedents."""
	if role not in ("prosecution", "defense"):
		raise ValueError("role must be 'prosecution' or 'defense'")
	if not facts or not facts.strip():
		raise ValueError("Facts must be a non-empty string.")

	print(f"[precedent_retriever] Running {role} role-specific retrieval...")

	# Step 1: Generate role-biased query variations
	variations = _generate_role_queries(facts, role)

	# Step 2: Retrieve for each variation using Hybrid search
	hybrid = _get_hybrid_retriever(top_k=top_k)

	all_results = []
	for i, variation in enumerate(variations, 1):
		docs = hybrid.invoke(variation)
		all_results.append(docs)
		print(f"[precedent_retriever] {role.capitalize()} query {i} returned {len(docs)} chunks.")

	# Step 3: Fuse with RRF
	fused = _reciprocal_rank_fusion(all_results, k=60)
	print(f"[precedent_retriever] RRF fused {len(fused)} unique chunks for {role}.")

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

	print(f"[precedent_retriever] ✅ {role.capitalize()} retrieval returned {len(results)} precedents.")
	return results


# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────

def _format_results(docs: list) -> List[Dict[str, object]]:
	"""Format LangChain Document objects into result dicts."""
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
	return results


if __name__ == "__main__":
	_build_vector_store()
