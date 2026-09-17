"""Run the Casebook agent over the golden dataset and score it with RAGAS."""

import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from graph_agent import ask_agent
from evaluation.ragas_config import get_ragas_embeddings, get_ragas_llm
from pipeline import load_vector_store


DATASET_PATH = Path("evaluation/golden_dataset.json")
OUTPUT_PATH = Path("evaluation/agent_outputs.json")
RESULTS_PATH = Path("evaluation/agent_ragas_results.csv")


def collect_outputs(vector_store) -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        golden_dataset = json.load(file)

    outputs = []
    total = len(golden_dataset)
    for index, item in enumerate(golden_dataset, start=1):
        question = item["question"]
        print(f"[{index}/{total}] {question}")
        result = ask_agent(question, vector_store)
        outputs.append(
            {
                "question": question,
                "ground_truth": item["ground_truth"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "tool_calls": result["tool_calls"],
            }
        )

    OUTPUT_PATH.write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Agent outputs saved to {OUTPUT_PATH}")
    return outputs


def run_ragas(outputs: list[dict]) -> None:
    dataset = Dataset.from_dict(
        {
            "question": [item["question"] for item in outputs],
            "answer": [item["answer"] for item in outputs],
            "contexts": [item["contexts"] for item in outputs],
            "ground_truth": [item["ground_truth"] for item in outputs],
        }
    )
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
    )
    print(result)
    result.to_pandas().to_csv(RESULTS_PATH, index=False)
    print(f"Per-query results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_ragas(collect_outputs(load_vector_store()))
