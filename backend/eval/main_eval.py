"""Entry point for Ragas evaluation of the Phase 2 Court RAG pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from retrieval import precedent_retriever, statute_retriever
from eval.run_eval import run_single_case, CASES_DIR, TEST_CASES
from eval.score import score


def main() -> None:
    print("=" * 55)
    print("COURT RAG — RAGAS EVALUATION")
    print(f"Test cases: {TEST_CASES}")
    print("=" * 55)

    # Initialize both retrievers (loads ChromaDB + builds BM25 index)
    print("\n[eval] Initializing retrievers...")
    statute_retriever.initialize_statute_retriever()
    precedent_retriever.initialize_precedent_retriever()

    samples = []
    for i, filename in enumerate(TEST_CASES, 1):
        case_path = CASES_DIR / filename
        if not case_path.exists():
            print(f"[eval] WARNING: {filename} not found — skipping.")
            continue

        print(f"\n[eval] ━━━ Case {i}/{len(TEST_CASES)}: {filename} ━━━")
        case_text = case_path.read_text(encoding="utf-8", errors="ignore")
        result = run_single_case(case_text)
        samples.append(result)
        print(f"[eval] ✅ Case {filename} complete.")

    if not samples:
        print("[eval] No valid cases found. Exiting.")
        return

    score(samples)


if __name__ == "__main__":
    main()
