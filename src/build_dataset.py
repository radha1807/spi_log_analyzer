
import pandas as pd
from pathlib import Path

from feature_extractor import extract_features


# define project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_LOG_DIR = PROJECT_ROOT / "data" / "raw_logs"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

METADATA_FILE = PROJECT_ROOT / "data" / "dataset_metadata.csv"

OUTPUT_FILE = PROCESSED_DIR / "spi_features.csv"


# create the processed directory if it does not exist
PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# extract features from all raw captures and attach ground truth labels
def build_dataset():

    capture_files = sorted(
        RAW_LOG_DIR.glob("capture_*.csv")
    )

    if not capture_files:

        raise FileNotFoundError(
            f"No SPI captures found in {RAW_LOG_DIR}"
        )

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    metadata = pd.read_csv(
        METADATA_FILE
    )

    required_metadata_columns = [
        "filename",
        "failure_type"
    ]

    missing_columns = [
        column
        for column in required_metadata_columns
        if column not in metadata.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing metadata columns: {missing_columns}"
        )

    records = []

    print(
        f"Found {len(capture_files)} SPI captures."
    )

    print(
        "Extracting features..."
    )

    for index, capture_file in enumerate(
        capture_files,
        start=1
    ):

        try:

            features = extract_features(
                capture_file
            )

            metadata_row = metadata[
                metadata["filename"]
                == capture_file.name
            ]

            if metadata_row.empty:

                print(
                    f"Warning: no metadata found for "
                    f"{capture_file.name}"
                )

                continue

            failure_type = (
                metadata_row.iloc[0]["failure_type"]
            )

            features["failure_type"] = (
                failure_type
            )

            records.append(
                features
            )

            if index % 50 == 0:

                print(
                    f"Processed "
                    f"{index}/{len(capture_files)} "
                    f"captures"
                )

        except Exception as error:

            print(
                f"Error processing "
                f"{capture_file.name}: "
                f"{error}"
            )

    if not records:

        raise RuntimeError(
            "No captures were successfully processed."
        )

    dataset = pd.DataFrame(
        records
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSuccessfully processed "
        f"{len(dataset)} captures."
    )

    print(
        f"Dataset saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nDataset shape: "
        f"{dataset.shape}"
    )

    print(
        "\nFailure distribution:"
    )

    print(
        dataset["failure_type"]
        .value_counts()
    )

    print(
        "\nFeature columns:"
    )

    for column in dataset.columns:

        print(
            f"- {column}"
        )


# run the dataset builder
if __name__ == "__main__":

    build_dataset()