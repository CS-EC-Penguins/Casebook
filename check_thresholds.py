"""
Exit non-zero if any RAGAS metric in a results JSON file falls below threshold.

This file is given complete: the loop is plumbing. The part that is yours is the
THRESHOLDS dictionary. Set each value roughly 0.10 to 0.15 below your own Week 9
baseline, so the gate catches a real regression without firing on judge variance.

Usage:
    python check_thresholds.py --results evaluation_results.json
"""
import argparse
import json
import sys

# Set your own thresholds here, roughly 15 percentage points below your
# Week 9 baseline scores. Adjust once you have established your own baseline.
THRESHOLDS = {
    "faithfulness": 0.65,
    "answer_relevancy": 0.65,
    "context_precision": 0.65,
    "context_recall": 0.55,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    failures = []
    for metric, threshold in THRESHOLDS.items():
        score = results.get(metric)
        if score is None:
            print(f"WARNING: {metric} not found in results")
            continue
        status = "PASS" if score >= threshold else "FAIL"
        print(f"{metric}: {score:.3f} (threshold: {threshold}) [{status}]")
        if score < threshold:
            failures.append(metric)

    if failures:
        print(f"\nCI FAILED: the following metrics are below threshold: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("\nAll metrics above threshold.")


if __name__ == "__main__":
    main()
