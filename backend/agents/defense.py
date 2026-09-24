from __future__ import annotations

from utils.llm import generate_response
from utils.prompts import get_defense_prompt, get_defense_rebuttal_prompt


def run_defense_agent(facts: str, prosecution_output: str, precedents: str) -> str:
	"""Round 1: Opening argument."""
	prompt = get_defense_prompt(facts=facts, prosecution_output=prosecution_output, precedents=precedents)
	return generate_response(prompt)


def run_defense_rebuttal(
	facts: str,
	precedents: str,
	own_opening: str,
	prosecution_opening: str,
	prosecution_rebuttal: str,
) -> str:
	"""Round 2: Rebuttal to prosecution's rebuttal."""
	prompt = get_defense_rebuttal_prompt(
		facts=facts,
		precedents=precedents,
		own_opening=own_opening,
		prosecution_opening=prosecution_opening,
		prosecution_rebuttal=prosecution_rebuttal,
	)
	return generate_response(prompt)
