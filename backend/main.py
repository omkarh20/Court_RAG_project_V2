from agents.defense import run_defense_agent
from agents.judge import run_judge_agent
from agents.prosecution import run_prosecution_agent
from retrieval import precedent_retriever, statute_retriever
from utils.llm import generate_response
from utils.prompts import get_fact_extraction_prompt


INPUT_CASE_TEXT = (
	"The accused struck the victim multiple times with an iron rod during a dispute. "
	"Witnesses stated the attack was intentional and directed at vital body parts. "
	"The victim later died from head injuries caused by the assault."
)


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


def _print_statutes(statutes: list[dict]) -> None:
	print("\n========== STATUTES ==========")
	if not statutes:
		print("No statutes retrieved.")
		return

	for statute in statutes:
		print(f"\nRank: {statute.get('rank')}")
		print(f"Statute: {statute.get('filename', '')}")
		print(f"Preview: {statute.get('text', '')[:200]}")


def _print_precedents(precedents: list[dict]) -> None:
	print("\n========== PRECEDENTS ==========")
	if not precedents:
		print("No precedents retrieved.")
		return

	for precedent in precedents:
		print(f"\nRank: {precedent.get('rank')}")
		print(f"Case ID: {precedent.get('case_id', '')}")
		print(f"Preview: {precedent.get('text_preview', '')[:200]}")


def run_legal_pipeline(case_text: str) -> None:
	if not case_text or not case_text.strip():
		raise ValueError("Input case text must be a non-empty string.")

	print("[main] Starting legal RAG pipeline...")

	# Step 1: Initialize vector databases
	statute_init = statute_retriever.initialize_statute_retriever()
	print(f"[main] Statutes loaded: {statute_init['statute_count']} chunks")

	precedent_init = precedent_retriever.initialize_precedent_retriever()
	print(f"[main] Cases loaded: {precedent_init['case_count']} chunks")

	# Step 2: Extract structured facts from raw case text
	print("[main] Extracting facts using LLM...")
	fact_prompt = get_fact_extraction_prompt(case_text)
	facts = generate_response(fact_prompt)

	# Step 3: Retrieve relevant statutes and precedents
	print("[main] Retrieving statutes based on extracted facts...")
	statutes = statute_retriever.retrieve_statutes(facts, top_k=5)

	print("[main] Retrieving precedents using multi-query + RRF...")
	precedents = precedent_retriever.retrieve_precedents_multi_query(facts, top_k=15)

	# Step 4: Format retrieved context for prompts
	statutes_for_prompt = _format_statutes_for_prompt(statutes)
	precedents_for_prompt = _format_precedents_for_prompt(precedents)

	# Step 5: Run the sequential multi-role prompt chain
	print("[main] Running prosecution agent...")
	prosecution_output = run_prosecution_agent(
		facts=facts,
		statutes=statutes_for_prompt,
		precedents=precedents_for_prompt,
	)

	print("[main] Running defense agent...")
	defense_output = run_defense_agent(
		facts=facts,
		prosecution_output=prosecution_output,
		precedents=precedents_for_prompt,
	)

	print("[main] Running judge agent...")
	judge_output = run_judge_agent(
		facts=facts,
		prosecution_output=prosecution_output,
		defense_output=defense_output,
		statutes=statutes_for_prompt,
	)

	# Step 6: Display results
	print("\n========== FACTS ==========")
	print(facts)

	_print_statutes(statutes)
	_print_precedents(precedents)

	print("\n========== PROSECUTION ARGUMENT ==========")
	print(prosecution_output)

	print("\n========== DEFENSE ARGUMENT ==========")
	print(defense_output)

	print("\n========== FINAL VERDICT ==========")
	print(judge_output)


if __name__ == "__main__":
	run_legal_pipeline(INPUT_CASE_TEXT)
