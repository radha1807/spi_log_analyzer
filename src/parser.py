import pandas as pd
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_LOG_DIR = PROJECT_ROOT / "data" / "raw_logs"


def load_spi_log(file_path):
    """
    Load a raw SPI CSV capture.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"SPI log not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "timestamp_us",
        "clk",
        "mosi",
        "miso"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


def get_basic_statistics(df):
    """
    Calculate basic statistics from an SPI capture.
    """

    duration_us = (
        df["timestamp_us"].max()
        - df["timestamp_us"].min()
    )

    mosi_changes = (
        df["mosi"]
        .astype(int)
        .diff()
        .abs()
        .sum()
    )

    miso_changes = (
        df["miso"]
        .astype(int)
        .diff()
        .abs()
        .sum()
    )

    clk_changes = (
        df["clk"]
        .astype(int)
        .diff()
        .abs()
        .sum()
    )

    return {
        "total_samples": len(df),
        "duration_us": duration_us,
        "mosi_changes": int(mosi_changes),
        "miso_changes": int(miso_changes),
        "clk_changes": int(clk_changes),
    }


def parse_capture(file_path):
    """
    Load and analyze one SPI capture.
    """

    df = load_spi_log(file_path)

    statistics = get_basic_statistics(df)

    return {
        "file": Path(file_path).name,
        **statistics
    }


if __name__ == "__main__":

    # Automatically select the first generated capture
    capture_files = sorted(
        RAW_LOG_DIR.glob("capture_*.csv")
    )

    if not capture_files:

        print(
            "No SPI captures found in:"
        )

        print(RAW_LOG_DIR)

        print(
            "\nRun generate_dataset.py first."
        )

        raise SystemExit(1)

    first_capture = capture_files[0]

    print("=" * 50)
    print("SPI LOG PARSER")
    print("=" * 50)

    print(f"\nAnalyzing:")
    print(first_capture)

    result = parse_capture(first_capture)

    print("\nResults:")
    print("-" * 50)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )