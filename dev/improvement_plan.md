# Court RAG Improvement Plan

Applying RAG techniques learned from `rag-for-beginners` to the Court RAG project.
Switching from Gemini + SentenceTransformers + FAISS → **OpenAI + LangChain + ChromaDB**.

---

## Current State of Court RAG

| Component | What it does now | Problem |
|---|---|---|
| **Chunking** | None. Each case file is one blob (~20K chars). | `all-MiniLM-L6-v2` truncates at ~256 tokens. ~95% of each case is invisible to search. |
| **Embeddings** | `all-MiniLM-L6-v2` (SentenceTransformers, 384 dims) | Small model, local-only, poor legal vocabulary. |
| **Vector Store** | FAISS + pickle files | ~220 lines of manual index/metadata management per retriever. |
| **Retrieval** | Blind top-k similarity | No score filtering, no diversity (MMR), no multi-query. |
| **Generation** | Raw `google.generativeai` SDK | No LangChain, manual retry logic, fragile. |
| **Agents** | Prompt templates pretending to be agents | No tool use, no memory, no chains. Precedents retrieved but NEVER passed to agents! |

## Target State

| Component | New approach |
|---|---|
| **Chunking** | `RecursiveCharacterTextSplitter` (800 chars, 100 overlap) |
| **Embeddings** | OpenAI `text-embedding-3-small` (1536 dims) via LangChain |
| **Vector Store** | ChromaDB with `persist_directory` (auto-save, built-in metadata) |
| **Retrieval** | Score threshold + MMR + Multi-query with RRF (fetch top 15 chunks) |
| **Generation** | `ChatOpenAI(model="gpt-4o-mini")` via LangChain |
| **Agents** | LangChain message format (`SystemMessage` + `HumanMessage`) |

## Cost Estimate (OpenAI Tier 1, ~$5 deposit)

| Operation | Tokens | Cost |
|---|---|---|
| Embed 73K case chunks (one-time) | ~14.6M tokens | **~$0.29** |
| Embed 985 statute chunks (one-time) | ~200K tokens | **~$0.004** |
| Per query (multi-query generation) | ~500 tokens | **~$0.0001** |
| Per query (3 agent responses) | ~5K tokens output | **~$0.003** |
| **Total for 500 test queries** | | **~$1.55** |
| **Grand total (build + 500 queries)** | | **~$1.85** |

---

## Improvement Plan (Phase 1: Architecture Upgrade)

### Task 1: Switch to OpenAI + ChromaDB + Add Chunking

This is the biggest task. We replace the entire retrieval backend.

**Files to delete:**
- `vector_db/cases.index`, `vector_db/cases.pkl` (old FAISS artifacts)
- `vector_db/statutes.index`, `vector_db/statutes.pkl`

**Files to change:**

#### `requirements.txt`
```diff
- sentence-transformers
- faiss-cpu
+ langchain
+ langchain-openai
+ langchain-chroma
+ langchain-text-splitters
+ langchain-community
```

#### `.env`
```diff
- GEMINI_API_KEY=...
+ OPENAI_API_KEY=sk-...
```

#### `backend/retrieval/precedent_retriever.py` (rewrite ~220 lines → ~60 lines)
- Remove all FAISS/pickle/numpy/SentenceTransformer code
- New flow:
  1. `DirectoryLoader` loads all `.txt` files from `data/cases/`
  2. `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)` splits them
  3. `Chroma.from_documents()` embeds and stores with `OpenAIEmbeddings(model="text-embedding-3-small")`
  4. `retrieve_precedents()` uses `db.as_retriever(search_kwargs={"k": 15})`

#### `backend/retrieval/statute_retriever.py` (rewrite ~280 lines → ~60 lines)
- Same approach as precedent retriever
- `chunk_size=500` (statutes are shorter)

#### `backend/utils/llm.py` (rewrite ~50 lines → ~20 lines)
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOpenAI(model="gpt-4o-mini")
```

---

### Task 2: Better Retrieval (Score Threshold + MMR)

Now that we have Chunking (creating ~73,000 chunks), we need to fetch more chunks (10-15) and ensure diversity using MMR.

**Files to change:**

#### `backend/retrieval/precedent_retriever.py`
```python
# With MMR (avoids getting 15 chunks from the exact same case)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": 0.5}
)
```

---

### Task 3: Multi-Query Retrieval + RRF

For complex legal queries, generate 3 query variations and merge results.

#### `backend/retrieval/precedent_retriever.py` — add `retrieve_precedents_multi_query()`
- Takes original query
- Calls `ChatOpenAI` to generate 3 legal query variations
- Runs `retrieve_precedents()` for each
- Merges with RRF (copy the function from `rag-for-beginners`)
- Returns fused top 15 chunks

---

### Task 4: Fix Precedent Data Leak (Critical Bug Fix)

The V1 codebase retrieves precedents but **never passes them to the Prosecution or Defense**.

#### `backend/main.py`
```python
# Create a formatter for precedents
precedents_for_prompt = _format_precedents_for_prompt(precedents)

# Pass precedents to the agents!
prosecution_output = run_prosecution_agent(
    facts=facts, statutes=statutes_for_prompt, precedents=precedents_for_prompt
)
```

#### `backend/utils/prompts.py`
- Update `get_prosecution_prompt` and `get_defense_prompt` to accept `precedents: str` and explicitly instruct the LLM to cite historical cases to back up their arguments.

---

### Task 5: Clean Up and Polish
- Remove dead code and old FAISS references
- Update `README.md` with new setup instructions
- Make sure `main.py` works end-to-end

---

## Execution Order (Phase 1)

```
Task 1 (OpenAI + Chroma + Chunking)  →  rebuild everything, test basic retrieval
Task 2 (Score Threshold + MMR)       →  one-line changes, test quality
Task 3 (Multi-Query + RRF)           →  add multi-query, test complex queries
Task 4 (Fix Precedent Data Leak)     →  update prompts to actually use case law
Task 5 (Clean Up)                    →  polish and document
```

---

## Phase 2: Advanced Multi-Agent System (Future Work)

Once Phase 1 is stable, we will transform the Sequential Chain into a fully-fledged Multi-Agent System with Tool Use.

1. **Independent Agentic Retrieval:**
   - Convert the Prosecution and Defense into LangChain Agents equipped with a `SearchChromaDB` tool. 
   - They will independently search for their own precedents (Prosecution looks for conviction precedents; Defense looks for acquittal/mitigation precedents).
2. **Adversarial Multi-Turn Debate:**
   - Replace the static chain with a stateful loop (like LangGraph). 
   - Allow Prosecution and Defense to cross-examine each other with rebuttals over 2-3 turns before handing the transcript to the Judge.
3. **Information Asymmetry (Blind Judge):**
   - The Judge agent will no longer receive the raw facts or statutes. It will *only* receive the final debate transcript presented by the lawyers.
4. **Structured JSON Verdicts:**
   - Use `.with_structured_output(PydanticModel)` to force the Judge to return a strict, easily-parsable JSON object containing `verdict`, `confidence_score`, and `sentence_years`. But also include the current way of response. Keep both.

---

## What We're NOT Doing (Yet)

- ❌ Semantic chunking (too API-heavy for 73K chunks)
- ❌ Agentic chunking (interesting concept but not practical for massive datasets)
- ❌ Multi-modal RAG (no images in court documents)
