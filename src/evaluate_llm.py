from pathlib import Path
import pandas as pd

from feature_extractor import extract_features
from retriever import retrieve_similar_captures
from llm_analyzer import (
    LLMAnalyzer,
    classify_target_features
)


# define project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_LOG_DIR = (
    PROJECT_ROOT /
    "data" /
    "raw_logs"
)

METADATA_PATH = (
    PROJECT_ROOT /
    "data" /
    "dataset_metadata.csv"
)


# load metadata
metadata = pd.read_csv(
    METADATA_PATH
)


# select one representative capture
# from every failure class
representative_rows = (
    metadata
    .sort_values("capture_id")
    .groupby(
        "failure_type",
        sort=True
    )
    .first()
    .reset_index()
)


print()
print("=" * 70)
print("LLM DIAGNOSIS EVALUATION")
print("=" * 70)

print()
print(
    f"Testing "
    f"{len(representative_rows)} "
    f"failure classes."
)

print()


analyzer = LLMAnalyzer()

results = []


for _, row in representative_rows.iterrows():

    failure_type = row["failure_type"]
    filename = row["filename"]

    capture_path = (
        RAW_LOG_DIR /
        filename
    )

    print("=" * 70)
    print(
        f"Failure class: "
        f"{failure_type}"
    )

    print(
        f"Capture: "
        f"{filename}"
    )

    print("=" * 70)

    try:

        # extract target features
        features = extract_features(
            capture_path
        )

        # determine deterministic diagnosis
        deterministic_type = (
            classify_target_features(
                features
            )
        )

        # retrieve historical evidence
        (
            retrieval_features,
            retrieval_document,
            retrieval_output
        ) = retrieve_similar_captures(
            capture_path,
            top_k=5
        )

        distances = (
            retrieval_output
            .get("distances", [[]])[0]
        )

        metadatas = (
            retrieval_output
            .get("metadatas", [[]])[0]
        )

        documents = (
            retrieval_output
            .get("documents", [[]])[0]
        )

        retrieved_results = []

        for index in range(
            len(distances)
        ):

            retrieved_results.append({
                "rank": index + 1,
                "distance": float(
                    distances[index]
                ),
                "metadata": (
                    metadatas[index]
                    if index < len(metadatas)
                    else {}
                ),
                "document": (
                    documents[index]
                    if index < len(documents)
                    else ""
                )
            })

        # run llm diagnosis
        diagnosis = analyzer.analyze(
            features,
            retrieved_results
        )

        llm_type = diagnosis[
            "failure_type"
        ]

        matches_expected = (
            llm_type ==
            failure_type
        )

        matches_deterministic = (
            llm_type ==
            deterministic_type
        )

        print()
        print(
            f"Expected:       "
            f"{failure_type}"
        )

        print(
            f"Deterministic:  "
            f"{deterministic_type}"
        )

        print(
            f"LLM:            "
            f"{llm_type}"
        )

        print(
            f"Confidence:     "
            f"{diagnosis['confidence']:.2f}"
        )

        print()
        print("Evidence:")

        for evidence in diagnosis[
            "evidence"
        ]:

            print(
                f"- {evidence}"
            )

        print()
        print(
            "Root cause:"
        )

        print(
            diagnosis[
                "probable_root_cause"
            ]
        )

        print()
        print(
            "Recommended checks:"
        )

        for check in diagnosis[
            "recommended_checks"
        ]:

            print(
                f"- {check}"
            )

        print()

        results.append({
            "failure_type": failure_type,
            "file": filename,
            "deterministic_type": (
                deterministic_type
            ),
            "llm_type": llm_type,
            "confidence": (
                diagnosis["confidence"]
            ),
            "matches_expected": (
                matches_expected
            ),
            "matches_deterministic": (
                matches_deterministic
            )
        })

    except Exception as error:

        print()
        print(
            f"ERROR: {error}"
        )

        results.append({
            "failure_type": failure_type,
            "file": filename,
            "deterministic_type": None,
            "llm_type": None,
            "confidence": None,
            "matches_expected": False,
            "matches_deterministic": False
        })

    print()


# convert results to dataframe
results_df = pd.DataFrame(
    results
)


# calculate llm accuracy
successful_results = (
    results_df[
        results_df["llm_type"].notna()
    ]
)

if len(successful_results) > 0:

    llm_accuracy = (
        successful_results[
            "matches_expected"
        ].mean()
    )

else:

    llm_accuracy = 0


print()
print("=" * 70)
print("LLM EVALUATION SUMMARY")
print("=" * 70)

print()

print(
    f"Failure classes tested: "
    f"{len(results_df)}"
)

print(
    f"Successful LLM analyses: "
    f"{len(successful_results)}"
)

print(
    f"LLM classification accuracy: "
    f"{llm_accuracy * 100:.2f}%"
)

print()

print(
    results_df[
        [
            "failure_type",
            "deterministic_type",
            "llm_type",
            "confidence",
            "matches_expected"
        ]
    ].to_string(
        index=False
    )
)


# save results
output_path = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "llm_evaluation.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


print()
print(
    f"Results saved to: "
    f"{output_path}"
)

print()