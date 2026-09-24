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


# ──────────────────────────────────────────────────────────────────
# ROUND 1: OPENING ARGUMENTS
# ──────────────────────────────────────────────────────────────────

def get_prosecution_prompt(facts: str, statutes: str, precedents: str) -> str:
	return (
		"You are the prosecution lawyer. This is your OPENING ARGUMENT.\n"
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
		"You are the defense lawyer. This is your OPENING ARGUMENT.\n"
		"Counter the prosecution argument and reduce criminal liability where possible.\n"
		"Cite historical precedent cases that support the defense's position.\n"
		"Be concise and structured with these sections:\n"
		"1) Weaknesses in Prosecution Case\n2) Alternative Interpretation\n3) Supporting Precedents\n4) Mitigating Factors\n5) Relief Sought\n\n"
		f"Facts:\n{facts.strip()}\n\n"
		f"Prosecution Argument:\n{prosecution_output.strip()}\n\n"
		f"Historical Precedents:\n{precedents.strip()}"
	)


# ──────────────────────────────────────────────────────────────────
# ROUND 2: REBUTTALS
# ──────────────────────────────────────────────────────────────────

def get_prosecution_rebuttal_prompt(
	facts: str,
	statutes: str,
	precedents: str,
	own_opening: str,
	opponent_opening: str,
) -> str:
	return (
		"You are the prosecution lawyer. This is your REBUTTAL.\n"
		"The defense has presented their opening argument below. You must:\n"
		"1) Directly counter the defense's weakest points\n"
		"2) Reinforce your strongest evidence\n"
		"3) Cite additional precedents that undermine the defense's position\n"
		"4) Conclude with a strong call for conviction\n\n"
		"Be forceful but precise. Address the defense's arguments point by point.\n\n"
		f"Facts:\n{facts.strip()}\n\n"
		f"Relevant Statutes:\n{statutes.strip()}\n\n"
		f"Your Opening Argument:\n{own_opening.strip()}\n\n"
		f"Defense's Opening Argument:\n{opponent_opening.strip()}\n\n"
		f"Your Precedents:\n{precedents.strip()}"
	)


def get_defense_rebuttal_prompt(
	facts: str,
	precedents: str,
	own_opening: str,
	prosecution_opening: str,
	prosecution_rebuttal: str,
) -> str:
	return (
		"You are the defense lawyer. This is your REBUTTAL.\n"
		"The prosecution has presented their rebuttal below. You must:\n"
		"1) Dismantle the prosecution's strongest claims\n"
		"2) Highlight any overreach or assumptions in their rebuttal\n"
		"3) Cite additional precedents supporting acquittal or reduced sentencing\n"
		"4) Make a final emotional and legal appeal for your client\n\n"
		"Be persuasive and thorough. This is your last chance to argue.\n\n"
		f"Facts:\n{facts.strip()}\n\n"
		f"Your Opening Argument:\n{own_opening.strip()}\n\n"
		f"Prosecution's Opening Argument:\n{prosecution_opening.strip()}\n\n"
		f"Prosecution's Rebuttal:\n{prosecution_rebuttal.strip()}\n\n"
		f"Your Precedents:\n{precedents.strip()}"
	)


# ──────────────────────────────────────────────────────────────────
# JUDGE (BLIND — ONLY SEES DEBATE TRANSCRIPT)
# ──────────────────────────────────────────────────────────────────

def get_judge_prompt(debate_transcript: str) -> str:
	return (
		"You are the judge presiding over this criminal case.\n"
		"You have NOT been briefed on the raw facts or statutes independently. "
		"Your ONLY source of information is the debate transcript below, "
		"which contains the opening arguments and rebuttals from both the "
		"prosecution and the defense.\n\n"
		"Based SOLELY on the arguments and evidence presented in the debate, "
		"deliver a reasoned verdict.\n"
		"Return in this structure:\n"
		"Findings:\n- ...\n"
		"Reasoning:\n- ...\n"
		"Final Verdict:\n- ...\n"
		"Suggested Sentence/Relief:\n- ...\n\n"
		f"Debate Transcript:\n{debate_transcript}"
	)
