"""Score collected pipeline outputs using Ragas (reference-free metrics)."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy

# ── Use gpt-4o-mini to keep costs low ──────────────────────────────
_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

METRICS = [faithfulness, answer_relevancy]

OUTPUT_DIR = Path(__file__).resolve().parent


def score(samples: list[dict]) -> None:
    """Run Ragas evaluation on a list of pipeline samples and print + save results."""
    print("\n[ragas] Building dataset...")
    dataset = Dataset.from_list(samples)

    print("[ragas] Running evaluation with gpt-4o-mini...")
    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=_llm,
        embeddings=_embeddings,
    )

    df = result.to_pandas()

    print("\n" + "=" * 55)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 55)
    cols = ["faithfulness", "answer_relevancy"]
    print(df[cols].round(3).to_string())

    print("\n" + "=" * 55)
    print("AVERAGES ACROSS ALL CASES")
    print("=" * 55)
    for col in cols:
        print(f"  {col:<25} {df[col].mean():.3f}")

    output_path = OUTPUT_DIR / "ragas_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[ragas] Full results saved to {output_path}")
