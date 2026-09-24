from agents.defense import run_defense_agent, run_defense_rebuttal
from agents.judge import run_judge_agent
from agents.prosecution import run_prosecution_agent, run_prosecution_rebuttal
from retrieval import precedent_retriever, statute_retriever
from utils.llm import generate_response
from utils.prompts import get_fact_extraction_prompt


INPUT_CASE_TEXT = (
	"The accused struck the victim multiple times with an iron rod during a dispute. "
	"Witnesses stated the attack was intentional and directed at vital body parts. "
	"The victim later died from head injuries caused by the assault."
)


# ──────────────────────────────────────────────────────────────────
# FORMATTERS
# ──────────────────────────────────────────────────────────────────

def _format_statutes_for_prompt(statutes: list[dict]) -> str:
	if not statutes:
		return "No statutes retrieved."
	parts = []
	for statute in statutes:
		filename = statute.get("filename", "")
		title = statute.get("title", "")
		text = (statute.get("text", "") or "").replace("\n", " ").strip()
		parts.append(f"- {filename} | {title}: {text[:500]}")
	return "\n".join(parts)


def _format_precedents_for_prompt(precedents: list[dict]) -> str:
	"""Format retrieved precedent chunks into a string for the LLM prompt."""
	if not precedents:
		return "No precedents retrieved."
	parts = []
	for precedent in precedents:
		case_id = precedent.get("case_id", "unknown")
		text = (precedent.get("text", "") or "").replace("\n", " ").strip()
		parts.append(f"- [{case_id}]: {text[:500]}")
	return "\n".join(parts)


def _format_debate_transcript(p_r1: str, d_r1: str, p_r2: str, d_r2: str) -> str:
	"""Build the full debate transcript that the blind judge will receive."""
	return (
		"=== ROUND 1: OPENING ARGUMENTS ===\n\n"
		f"PROSECUTION OPENING:\n{p_r1}\n\n"
		f"DEFENSE OPENING:\n{d_r1}\n\n"
		"=== ROUND 2: REBUTTALS ===\n\n"
		f"PROSECUTION REBUTTAL:\n{p_r2}\n\n"
		f"DEFENSE REBUTTAL:\n{d_r2}"
	)


# ──────────────────────────────────────────────────────────────────
# PRINTERS
# ──────────────────────────────────────────────────────────────────

def _print_statutes(statutes: list[dict]) -> None:
	print("\n========== STATUTES ==========")
	if not statutes:
		print("No statutes retrieved.")
		return

	for statute in statutes:
		print(f"\nRank: {statute.get('rank')}")
		print(f"Statute: {statute.get('filename', '')}")
		print(f"Preview: {statute.get('text', '')[:200]}")


def _print_precedents(precedents: list[dict], role: str) -> None:
	print(f"\n========== {role.upper()} PRECEDENTS ==========")
	if not precedents:
		print("No precedents retrieved.")
		return

	for precedent in precedents:
		print(f"\nRank: {precedent.get('rank')}")
		print(f"Case ID: {precedent.get('case_id', '')}")
		print(f"Preview: {precedent.get('text_preview', '')[:200]}")


# ──────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────

def run_legal_pipeline(case_text: str) -> None:
	if not case_text or not case_text.strip():
		raise ValueError("Input case text must be a non-empty string.")

	print("[main] Starting legal RAG pipeline (Phase 2: Adversarial Debate)...")

	# ── Step 1: Initialize vector databases ──
	statute_init = statute_retriever.initialize_statute_retriever()
	print(f"[main] Statutes loaded: {statute_init['statute_count']} chunks")

	precedent_init = precedent_retriever.initialize_precedent_retriever()
	print(f"[main] Cases loaded: {precedent_init['case_count']} chunks")

	# ── Step 2: Extract structured facts ──
	print("[main] Extracting facts using LLM...")
	fact_prompt = get_fact_extraction_prompt(case_text)
	facts = generate_response(fact_prompt)

	# ── Step 3: Retrieve statutes (shared) ──
	print("[main] Retrieving statutes (Hybrid Search)...")
	statutes = statute_retriever.retrieve_statutes(facts, top_k=5)
	statutes_for_prompt = _format_statutes_for_prompt(statutes)

	# ── Step 4: Independent agentic retrieval ──
	print("[main] Prosecution is searching for conviction precedents...")
	prosecution_precedents = precedent_retriever.retrieve_for_role(facts, "prosecution", top_k=10)

	print("[main] Defense is searching for acquittal/mitigation precedents...")
	defense_precedents = precedent_retriever.retrieve_for_role(facts, "defense", top_k=10)

	prosecution_prec_prompt = _format_precedents_for_prompt(prosecution_precedents)
	defense_prec_prompt = _format_precedents_for_prompt(defense_precedents)

	# ── Step 5: Round 1 — Opening Arguments ──
	print("[main] === ROUND 1: OPENING ARGUMENTS ===")

	print("[main] Prosecution presenting opening argument...")
	prosecution_r1 = run_prosecution_agent(
		facts=facts,
		statutes=statutes_for_prompt,
		precedents=prosecution_prec_prompt,
	)

	print("[main] Defense presenting opening argument...")
	defense_r1 = run_defense_agent(
		facts=facts,
		prosecution_output=prosecution_r1,
		precedents=defense_prec_prompt,
	)

	# ── Step 6: Round 2 — Rebuttals ──
	print("[main] === ROUND 2: REBUTTALS ===")

	print("[main] Prosecution delivering rebuttal...")
	prosecution_r2 = run_prosecution_rebuttal(
		facts=facts,
		statutes=statutes_for_prompt,
		precedents=prosecution_prec_prompt,
		own_opening=prosecution_r1,
		opponent_opening=defense_r1,
	)

	print("[main] Defense delivering rebuttal...")
	defense_r2 = run_defense_rebuttal(
		facts=facts,
		precedents=defense_prec_prompt,
		own_opening=defense_r1,
		prosecution_opening=prosecution_r1,
		prosecution_rebuttal=prosecution_r2,
	)

	# ── Step 7: Build debate transcript ──
	debate_transcript = _format_debate_transcript(
		prosecution_r1, defense_r1, prosecution_r2, defense_r2
	)

	# ── Step 8: Blind Judge ──
	print("[main] === JUDGE DELIBERATION (Blind — transcript only) ===")
	judge_output = run_judge_agent(debate_transcript=debate_transcript)

	# ── Step 9: Display results ──
	print("\n" + "=" * 60)
	print("COURT RAG — ADVERSARIAL DEBATE RESULTS")
	print("=" * 60)

	print("\n========== FACTS ==========")
	print(facts)

	_print_statutes(statutes)
	_print_precedents(prosecution_precedents, "prosecution")
	_print_precedents(defense_precedents, "defense")

	print("\n========== ROUND 1: PROSECUTION OPENING ==========")
	print(prosecution_r1)

	print("\n========== ROUND 1: DEFENSE OPENING ==========")
	print(defense_r1)

	print("\n========== ROUND 2: PROSECUTION REBUTTAL ==========")
	print(prosecution_r2)

	print("\n========== ROUND 2: DEFENSE REBUTTAL ==========")
	print(defense_r2)

	print("\n========== FINAL VERDICT (BLIND JUDGE) ==========")
	print(judge_output)


if __name__ == "__main__":
	run_legal_pipeline(INPUT_CASE_TEXT)
