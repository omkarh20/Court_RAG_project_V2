from __future__ import annotations

from utils.llm import generate_response
from utils.prompts import get_prosecution_prompt, get_prosecution_rebuttal_prompt


def run_prosecution_agent(facts: str, statutes: str, precedents: str) -> str:
	"""Round 1: Opening argument."""
	prompt = get_prosecution_prompt(facts=facts, statutes=statutes, precedents=precedents)
	return generate_response(prompt)


def run_prosecution_rebuttal(
	facts: str,
	statutes: str,
	precedents: str,
	own_opening: str,
	opponent_opening: str,
) -> str:
	"""Round 2: Rebuttal to defense's opening argument."""
	prompt = get_prosecution_rebuttal_prompt(
		facts=facts,
		statutes=statutes,
		precedents=precedents,
		own_opening=own_opening,
		opponent_opening=opponent_opening,
	)
	return generate_response(prompt)
