from pathlib import Path
import pandas as pd

from feature_extractor import extract_features
from llm_analyzer import classify_target_features


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


# load dataset metadata
metadata = pd.read_csv(
    METADATA_PATH
)


results = []

print("Evaluating deterministic diagnosis...")
print()


for index, row in metadata.iterrows():

    capture_path = (
        RAW_LOG_DIR /
        row["filename"]
    )

    features = extract_features(
        capture_path
    )

    predicted_failure = (
        classify_target_features(
            features
        )
    )

    actual_failure = (
        row["failure_type"]
    )

    results.append({
        "file": row["filename"],
        "actual_failure": actual_failure,
        "predicted_failure": predicted_failure,
        "correct": (
            actual_failure ==
            predicted_failure
        )
    })

    if (index + 1) % 50 == 0:

        print(
            f"Evaluated "
            f"{index + 1}/"
            f"{len(metadata)} captures"
        )


results_df = pd.DataFrame(
    results
)


accuracy = (
    results_df["correct"].mean()
)


print()
print("=" * 60)
print("SPI DIAGNOSIS EVALUATION")
print("=" * 60)

print()
print(
    f"Total captures: "
    f"{len(results_df)}"
)

print(
    f"Correct predictions: "
    f"{results_df['correct'].sum()}"
)

print(
    f"Accuracy: "
    f"{accuracy * 100:.2f}%"
)


print()
print("Per-class results:")
print()


class_results = (
    results_df
    .groupby("actual_failure")
    .agg(
        total=("correct", "count"),
        correct=("correct", "sum")
    )
)

class_results["accuracy"] = (
    class_results["correct"]
    / class_results["total"]
    * 100
)

print(
    class_results
    .round(2)
    .to_string()
)


# display incorrect predictions
incorrect = results_df[
    ~results_df["correct"]
]


print()
print(
    f"Incorrect predictions: "
    f"{len(incorrect)}"
)

if len(incorrect) > 0:

    print()
    print("Incorrect cases:")

    print(
        incorrect.to_string(
            index=False
        )
    )


# save results
output_path = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "diagnosis_evaluation.csv"
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