"""LLM utility using OpenAI via LangChain."""

from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key or api_key.startswith("sk-your"):
	print("[llm] WARNING: OPENAI_API_KEY is missing or invalid in .env")

_model = None


def _get_model() -> ChatOpenAI:
	global _model
	if _model is not None:
		return _model

	print("[llm] Loading model: gpt-4o-mini")
	_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
	return _model


def generate_response(prompt: str, system_prompt: str = None) -> str:
	"""Generate a response using OpenAI via LangChain."""
	if not prompt or not prompt.strip():
		raise ValueError("Prompt must be a non-empty string.")

	model = _get_model()

	messages = []
	if system_prompt:
		messages.append(SystemMessage(content=system_prompt))
	messages.append(HumanMessage(content=prompt))

	try:
		response = model.invoke(messages)
		return response.content.strip()
	except Exception as error:
		return f"[llm] Generation failed: {error}"
