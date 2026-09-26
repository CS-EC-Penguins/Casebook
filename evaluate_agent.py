"""Run the Casebook agent over the golden dataset and score it with RAGAS."""
import argparse
import math
import json
from pathlib import Path
import sys

from datasets import Dataset
from ragas import evaluate, RunConfig
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
OUTPUT_PATH = Path("evaluation/pipeline_outputs.json")
RESULTS_PATH = Path("evaluation/ragas_results.csv")


def collect_outputs(vector_store, dataset_path: Path = DATASET_PATH) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8") as file:
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

    run_config = RunConfig(
            timeout=700,
            max_workers=16,
            log_tenacity=True,
        )
    
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        run_config=run_config,
        raise_exceptions=True,
    )

    df = result.to_pandas()
    df.to_csv(RESULTS_PATH, index=False)   # save first so the failing rows can be inspected
    print(f"Per-query results saved to {RESULTS_PATH}")
    
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    nan_rows = df[df[metric_cols].isna().any(axis=1)]
    if not nan_rows.empty:
        for _, row in nan_rows.iterrows():
            bad = [m for m in metric_cols if math.isnan(row[m])]
            print(f"::error::NaN score for {bad} on question: {row.get('user_input', row.get('question'))}")
        sys.exit(1)

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="Write results to evaluation_results.json")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Path to the question set to evaluate")
    args = parser.parse_args()

    vector_store = load_vector_store()
    outputs = collect_outputs(vector_store, args.dataset)
    df = run_ragas(outputs).to_pandas()
    
    results = {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "context_recall": float(df["context_recall"].mean()),
    }

    if args.ci:
        with open("evaluation_results.json", "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
        print("Results written to evaluation_results.json")
    else:
        for metric, score in results.items():
            print(f"{metric}: {score:.3f}")

if __name__ == "__main__":
    main()
