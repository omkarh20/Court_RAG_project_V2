"""Run the full Phase 2 pipeline on test cases and collect (question, contexts, answer) tuples."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend is on the path when run from any directory
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from retrieval import precedent_retriever, statute_retriever
from utils.llm import generate_response
from utils.prompts import get_fact_extraction_prompt
from agents.prosecution import run_prosecution_agent, run_prosecution_rebuttal
from agents.defense import run_defense_agent, run_defense_rebuttal
from agents.judge import run_judge_agent
from main import (
    _format_statutes_for_prompt,
    _format_precedents_for_prompt,
    _format_debate_transcript,
)

CASES_DIR = Path(__file__).resolve().parents[2] / "data" / "cases"

# 5 real case files from our existing dataset
TEST_CASES = ["C1.txt", "C10.txt", "C100.txt", "C1000.txt", "C1001.txt"]


def run_single_case(case_text: str) -> dict:
    """Run the full pipeline for one case and return a Ragas-compatible sample."""
    print("[eval] Extracting facts...")
    facts = generate_response(get_fact_extraction_prompt(case_text))

    print("[eval] Retrieving statutes...")
    statutes = statute_retriever.retrieve_statutes(facts, top_k=5)
    statutes_str = _format_statutes_for_prompt(statutes)

    print("[eval] Running role-specific retrieval...")
    prosecution_precedents = precedent_retriever.retrieve_for_role(facts, "prosecution", top_k=8)
    defense_precedents = precedent_retriever.retrieve_for_role(facts, "defense", top_k=8)
    prec_prec_str = _format_precedents_for_prompt(prosecution_precedents)
    def_prec_str = _format_precedents_for_prompt(defense_precedents)

    print("[eval] Running Round 1: Opening arguments...")
    p_r1 = run_prosecution_agent(facts, statutes_str, prec_prec_str)
    d_r1 = run_defense_agent(facts, p_r1, def_prec_str)

    print("[eval] Running Round 2: Rebuttals...")
    p_r2 = run_prosecution_rebuttal(facts, statutes_str, prec_prec_str, p_r1, d_r1)
    d_r2 = run_defense_rebuttal(facts, def_prec_str, d_r1, p_r1, p_r2)

    print("[eval] Running blind judge...")
    transcript = _format_debate_transcript(p_r1, d_r1, p_r2, d_r2)
    verdict = run_judge_agent(debate_transcript=transcript)

    # Combine prosecution + defense precedents as the full context Ragas will evaluate
    all_contexts = [p["text"] for p in prosecution_precedents + defense_precedents]

    return {
        "user_input": case_text[:600].strip(),   # Ragas: the question
        "retrieved_contexts": all_contexts,       # Ragas: the retrieved context
        "response": verdict,                      # Ragas: the generated answer
    }
