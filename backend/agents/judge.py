from __future__ import annotations

from utils.llm import generate_response
from utils.prompts import get_judge_prompt


def run_judge_agent(debate_transcript: str) -> str:
	"""Blind judge — only receives the debate transcript, not raw facts or statutes."""
	prompt = get_judge_prompt(debate_transcript=debate_transcript)
	return generate_response(prompt)
