from __future__ import annotations


def get_fact_extraction_prompt(case_text: str) -> str:
	return (
		"You are a legal fact extractor. Extract structured facts from the case text.\n"
		"Return only these fields in the exact format:\n"
		"act: ...\n"
		"intent: ...\n"
		"weapon: ...\n"
		"outcome: ...\n\n"
		f"Case text:\n{case_text.strip()}"
	)


def get_prosecution_prompt(facts: str, statutes: str, precedents: str) -> str:
	return (
		"You are the prosecution lawyer.\n"
		"Argue why the accused is guilty based on the facts, relevant statutes, and historical case precedents.\n"
		"Cite specific statutes and precedent cases to support your argument.\n"
		"Be concise and structured with these sections:\n"
		"1) Charges\n2) Key Evidence\n3) Statutory Basis\n4) Supporting Precedents\n5) Conclusion\n\n"
		f"Facts:\n{facts.strip()}\n\n"
		f"Relevant Statutes:\n{statutes.strip()}\n\n"
		f"Historical Precedents:\n{precedents.strip()}"
	)


def get_defense_prompt(facts: str, prosecution_output: str, precedents: str) -> str:
	return (
		"You are the defense lawyer.\n"
		"Counter the prosecution argument and reduce criminal liability where possible.\n"
		"Cite historical precedent cases that support the defense's position.\n"
		"Be concise and structured with these sections:\n"
		"1) Weaknesses in Prosecution Case\n2) Alternative Interpretation\n3) Supporting Precedents\n4) Mitigating Factors\n5) Relief Sought\n\n"
		f"Facts:\n{facts.strip()}\n\n"
		f"Prosecution Argument:\n{prosecution_output.strip()}\n\n"
		f"Historical Precedents:\n{precedents.strip()}"
	)


def get_judge_prompt(
	facts: str,
	prosecution_output: str,
	defense_output: str,
	statutes: str,
) -> str:
	return (
		"You are the judge. Compare prosecution and defense arguments and give a reasoned verdict.\n"
		"Return in this structure:\n"
		"Findings:\n- ...\n"
		"Reasoning:\n- ...\n"
		"Final Verdict:\n- ...\n"
		"Suggested Sentence/Relief:\n- ...\n\n"
		f"Facts:\n{facts.strip()}\n\n"
		f"Relevant Statutes:\n{statutes.strip()}\n\n"
		f"Prosecution Argument:\n{prosecution_output.strip()}\n\n"
		f"Defense Argument:\n{defense_output.strip()}"
	)
