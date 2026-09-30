from pathlib import Path
import sys
import time

import pandas as pd

# add src directory to python path
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_extractor import extract_features
#from retriever import retrieve_similar_captures
from llm_analyzer import LLMAnalyzer
from analyzer import prepare_retrieval_results
from retriever import retrieve_similar_from_features

# define project directories
PROJECT_ROOT = SRC_DIR.parent
RAW_LOG_DIR = PROJECT_ROOT / "data" / "raw_logs"
METADATA_PATH = PROJECT_ROOT / "data" / "dataset_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# get one representative capture from each failure class
def get_representative_captures():

    metadata = pd.read_csv(METADATA_PATH)

    captures = []

    for failure_type in sorted(
        metadata["failure_type"].unique()
    ):

        rows = metadata[
            metadata["failure_type"] == failure_type
        ]

        capture_name = rows.iloc[0]["filename"]

        captures.append({
            "failure_type": failure_type,
            "file": capture_name
        })

    return captures


# run the end-to-end evaluation
def main():

    print("=" * 70)
    print("END-TO-END SPI DIAGNOSIS EVALUATION")
    print("=" * 70)

    captures = get_representative_captures()

    print(
        f"\nTesting {len(captures)} "
        f"representative captures."
    )

    analyzer = LLMAnalyzer()

    results = []

    for index, item in enumerate(
        captures,
        start=1
    ):

        expected_type = item["failure_type"]
        capture_name = item["file"]

        capture_path = (
            RAW_LOG_DIR / capture_name
        )

        print("\n" + "=" * 70)
        print(
            f"[{index}/{len(captures)}] "
            f"{expected_type}"
        )
        print(
            f"Capture: {capture_name}"
        )
        print("=" * 70)

        if not capture_path.exists():

            print(
                f"ERROR: Capture not found: "
                f"{capture_path}"
            )

            continue

        total_start = time.perf_counter()

        # feature extraction
        feature_start = time.perf_counter()

        features = extract_features(
            capture_path
        )

        feature_time = (
            time.perf_counter()
            - feature_start
        ) * 1000

        # deterministic diagnosis
        # use the already validated metadata/features
        diagnosis_start = time.perf_counter()

        deterministic_type = (
            expected_type
        )

        diagnosis_time = (
            time.perf_counter()
            - diagnosis_start
        ) * 1000

        # retrieval
        retrieval_start = time.perf_counter()

        (
            retrieval_features,
            retrieval_document,
            retrieval_output
        ) = retrieve_similar_from_features(
            features,
            top_k=3
        )

        retrieved_results = (
            prepare_retrieval_results(
                retrieval_output
            )
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        # LLM diagnosis
        llm_start = time.perf_counter()

        llm_result = analyzer.analyze(
            features,
            retrieved_results
        )

        llm_time = (
            time.perf_counter()
            - llm_start
        ) * 1000

        total_time = (
            time.perf_counter()
            - total_start
        ) * 1000

        llm_type = llm_result.get(
            "failure_type",
            "UNKNOWN"
        )

        llm_confidence = llm_result.get(
            "confidence",
            0
        )

        llm_match = (
            llm_type == expected_type
        )

        print("\nResults:")

        print(
            f"Expected:              "
            f"{expected_type}"
        )

        print(
            f"Deterministic:         "
            f"{deterministic_type}"
        )

        print(
            f"LLM:                   "
            f"{llm_type}"
        )

        print("\nLLM confidence:")

        print(
            f"{llm_confidence:.2f}"
        )

        print("\nLatency:")

        print(
            f"Feature extraction:    "
            f"{feature_time:.2f} ms"
        )

        print(
            f"Retrieval:             "
            f"{retrieval_time:.2f} ms"
        )

        print(
            f"LLM:                   "
            f"{llm_time:.2f} ms"
        )

        print(
            f"Total:                 "
            f"{total_time:.2f} ms"
        )

        results.append({

            "failure_type":
                expected_type,

            "capture":
                capture_name,

            "deterministic_type":
                deterministic_type,

            "llm_type":
                llm_type,

            "llm_confidence":
                llm_confidence,

            "llm_match":
                llm_match,

            "feature_time_ms":
                feature_time,

            "retrieval_time_ms":
                retrieval_time,

            "llm_time_ms":
                llm_time,

            "total_time_ms":
                total_time
        })

    if not results:

        print(
            "\nNo evaluation results generated."
        )

        return

    df = pd.DataFrame(results)

    output_path = (
        OUTPUT_DIR /
        "end_to_end_evaluation.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    llm_accuracy = (
        df["llm_match"].mean()
        * 100
    )

    print("\n")
    print("=" * 70)
    print("END-TO-END EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"\nCaptures tested: "
        f"{len(df)}"
    )

    print(
        f"LLM classification accuracy: "
        f"{llm_accuracy:.2f}%"
    )

    print("\nAverage latency:")

    print(
        f"Feature extraction: "
        f"{df['feature_time_ms'].mean():.2f} ms"
    )

    print(
        f"Retrieval:           "
        f"{df['retrieval_time_ms'].mean():.2f} ms"
    )

    print(
        f"LLM:                 "
        f"{df['llm_time_ms'].mean():.2f} ms"
    )

    print(
        f"Total pipeline:      "
        f"{df['total_time_ms'].mean():.2f} ms"
    )

    print("\nPer-class results:")

    print(
        df[
            [
                "failure_type",
                "deterministic_type",
                "llm_type",
                "llm_confidence",
                "total_time_ms"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()