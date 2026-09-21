# evaluate.py
# The evaluation harness. Runs every golden-dataset question through the
# pipeline, then scores the results with RAGAS. Run with: python evaluate.py
#
# Baseline scores (fill in after your first run):
# faithfulness:
# answer_relevancy:
# context_precision:
# context_recall:
# Date:
import argparse
import json
from pathlib import Path
from datasets import Dataset
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from pipeline import ask, load_vector_store
from evaluation.ragas_config import get_ragas_llm, get_ragas_embeddings

DATASET_PATH = Path("evaluation/golden_dataset.json")
OUTPUT_PATH  = Path("evaluation/pipeline_outputs.json")


def collect_outputs(vector_store) -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        golden_dataset = json.load(file)

    outputs = []
    total = len(golden_dataset)

    for index, item in enumerate(golden_dataset, start=1):
        question = item["question"]
        print(f"[{index}/{total}] {question}")

        result = ask(question, vector_store, use_rewriting=True)

        outputs.append(
            {
                "question": question,
                "ground_truth": item["ground_truth"],
                "answer": result["answer"],
                "contexts": result["contexts"],
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(outputs, file, indent=2, ensure_ascii=False)

    print(f"Pipeline outputs saved to {OUTPUT_PATH}")
    return outputs


def run_ragas(outputs: list[dict]) -> None:
    """Score the collected outputs with RAGAS (given, no changes needed).

    Builds a HuggingFace Dataset from outputs and calls ragas.evaluate() with
    four metrics, using the Vertex AI judge and embeddings from ragas_config.
    Prints the aggregate scores and saves per-query results to CSV.
    """
    data = {
        "question":     [item["question"] for item in outputs],
        "answer":       [item["answer"] for item in outputs],
        "contexts":     [item["contexts"] for item in outputs],
        "ground_truth": [item["ground_truth"] for item in outputs],
    }
    dataset = Dataset.from_dict(data)
    
    run_config = RunConfig(
        timeout=300,
        max_retries=6,
        max_wait=30,
        max_workers=4,
        log_tenacity=True,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        run_config=run_config,
    )

    #print(result)
    result.to_pandas().to_csv("evaluation/ragas_results.csv", index=False)
    print("Per-query results saved to evaluation/ragas_results.csv")
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="Write results to evaluation_results.json")
    parser.add_argument("--dataset", default="evaluation/golden_dataset.json", help="Path to the question set to evaluate")
    args = parser.parse_args()

    # ...your existing evaluation code, loading questions from args.dataset
    # instead of a hardcoded path...

    vector_store = load_vector_store()
    outputs = collect_outputs(vector_store)
    result = run_ragas(outputs)
    
    # In Week 9 your harness produced a RAGAS result object; you convert it with
    # result.to_pandas() to write the CSV. Take the per-metric means from that DataFrame:
    df = result.to_pandas()
    results = {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "context_recall": float(df["context_recall"].mean()),
    }

    if args.ci:
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Results written to evaluation_results.json")
    else:
        for metric, score in results.items():
            print(f"{metric}: {score:.3f}")


    
