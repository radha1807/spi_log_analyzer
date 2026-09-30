from pathlib import Path
import statistics
import pandas as pd


# define project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# load one spi capture
def load_capture(file_path):
    file_path = Path(file_path)

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
            f"Missing columns: {missing_columns}"
        )

    return df


# convert eight bits into a byte
def bits_to_byte(bits):
    if len(bits) != 8:
        return None

    value = 0

    for index, bit in enumerate(bits):
        value |= int(bit) << index

    return value


# reverse bit order inside a byte
def reverse_bits(value):
    if value is None:
        return None

    binary = f"{value:08b}"

    return int(binary[::-1], 2)


# detect rising clock edges
def get_rising_edges(df):
    rising_edges = []

    if len(df) == 0:
        return rising_edges

    previous_clk = int(df.iloc[0]["clk"])

    if previous_clk == 1:
        rising_edges.append(df.iloc[0])

    for index in range(1, len(df)):
        current_clk = int(df.iloc[index]["clk"])

        if previous_clk == 0 and current_clk == 1:
            rising_edges.append(df.iloc[index])

        previous_clk = current_clk

    return rising_edges


# calculate rising-edge periods
def get_clock_periods(df):
    rising_edges = get_rising_edges(df)

    if len(rising_edges) < 2:
        return []

    timestamps = [
        float(row["timestamp_us"])
        for row in rising_edges
    ]

    periods = []

    for index in range(1, len(timestamps)):
        period = (
            timestamps[index]
            - timestamps[index - 1]
        )

        if period > 0:
            periods.append(period)

    return periods


# calculate clock frequency
def calculate_clock_frequency(df):
    periods = get_clock_periods(df)

    if not periods:
        return None

    median_period_us = statistics.median(periods)

    if median_period_us <= 0:
        return None

    return 1000 / median_period_us


# calculate clock period
def calculate_clock_period(df):
    periods = get_clock_periods(df)

    if not periods:
        return None

    return statistics.median(periods)


# extract command and response bytes
def extract_bytes(df):
    rising_edges = get_rising_edges(df)

    mosi_bits = [
        int(row["mosi"])
        for row in rising_edges
    ]

    miso_bits = [
        int(row["miso"])
        for row in rising_edges
    ]

    command = None
    observed_response = None

    # command occupies the first eight clock cycles
    if len(mosi_bits) >= 8:
        command = bits_to_byte(
            mosi_bits[:8]
        )

    # response starts after the command
    response_bits = miso_bits[8:]

    # response must contain at least eight bits
    if len(response_bits) >= 8:
        observed_response = bits_to_byte(
            response_bits[:8]
        )

    return command, observed_response


# calculate response frame length
def calculate_response_frame_length(df):
    rising_edges = get_rising_edges(df)

    if len(rising_edges) <= 8:
        return 0

    return len(rising_edges) - 8


# count signal transitions
def count_transitions(series):
    if len(series) < 2:
        return 0

    values = series.astype(int).tolist()

    transitions = 0

    for index in range(1, len(values)):
        if values[index] != values[index - 1]:
            transitions += 1

    return transitions


# get response bits
def get_response_bits(df):
    rising_edges = get_rising_edges(df)

    if len(rising_edges) <= 8:
        return []

    return [
        int(row["miso"])
        for row in rising_edges[8:]
    ]


# detect stuck-high miso
def detect_miso_stuck_high(df):
    response_bits = get_response_bits(df)

    if len(response_bits) != 8:
        return 0

    return int(
        all(bit == 1 for bit in response_bits)
    )


# detect stuck-low miso
def detect_miso_stuck_low(df):
    response_bits = get_response_bits(df)

    if len(response_bits) != 8:
        return 0

    return int(
        all(bit == 0 for bit in response_bits)
    )


# detect clock glitches
def detect_clock_glitch(df):
    periods = get_clock_periods(df)

    if len(periods) < 9:
        return 0

    # use the command frame as the normal clock reference
    command_periods = periods[:7]

    if not command_periods:
        return 0

    normal_period = statistics.median(
        command_periods
    )

    if normal_period <= 0:
        return 0

    # inspect periods inside the response frame
    response_periods = periods[8:]

    for period in response_periods:
        if period < normal_period * 0.5:
            return 1

    return 0


# calculate response delay
def calculate_response_delay(df):
    rising_edges = get_rising_edges(df)

    # no response frame
    if len(rising_edges) <= 8:
        return None

    command_end_timestamp = float(
        rising_edges[7]["timestamp_us"]
    )

    response_start_timestamp = float(
        rising_edges[8]["timestamp_us"]
    )

    delay = (
        response_start_timestamp
        - command_end_timestamp
    )

    # normal clock period
    periods = get_clock_periods(df)

    if not periods:
        return 0

    normal_period = statistics.median(
        periods[:7]
    )

    # remove the normal clock period
    response_delay = delay - normal_period

    if response_delay <= 1:
        return 0

    return response_delay


# detect timeout
def detect_timeout(df):
    response_frame_length = (
        calculate_response_frame_length(df)
    )

    # timeout captures contain substantially more
    # activity than a normal response frame
    if response_frame_length >= 12:
        return 1

    return 0


# detect bit-order error
def detect_bit_order_error(
    observed_response,
    expected_response
):
    if (
        observed_response is None
        or expected_response is None
    ):
        return 0

    reversed_expected = reverse_bits(
        expected_response
    )

    return int(
        observed_response == reversed_expected
    )


# extract all features from one capture
def extract_features(file_path):

    file_path = Path(file_path)

    df = load_capture(file_path)

    rising_edges = get_rising_edges(df)

    command, observed_response = extract_bytes(
        df
    )

    expected_response = 0xAA

    response_frame_length = (
        calculate_response_frame_length(df)
    )

    response_match = 0

    if observed_response is not None:
        response_match = int(
            observed_response == expected_response
        )

    bit_order_error = detect_bit_order_error(
        observed_response,
        expected_response
    )

    miso_stuck_high = detect_miso_stuck_high(
        df
    )

    miso_stuck_low = detect_miso_stuck_low(
        df
    )

    clock_glitch = detect_clock_glitch(
        df
    )

    response_delay_us = calculate_response_delay(
        df
    )

    timeout_detected = detect_timeout(
        df
    )

    miso_transition_count = count_transitions(
        df["miso"]
    )

    mosi_transition_count = count_transitions(
        df["mosi"]
    )

    clock_frequency_khz = (
        calculate_clock_frequency(df)
    )

    clock_period_us = (
        calculate_clock_period(df)
    )

    duration_us = None

    if len(df) > 1:
        duration_us = (
            float(df.iloc[-1]["timestamp_us"])
            - float(df.iloc[0]["timestamp_us"])
        )

    return {
        "file": file_path.name,
        "total_samples": len(df),
        "duration_us": duration_us,
        "clock_frequency_khz": clock_frequency_khz,
        "clock_period_us": clock_period_us,
        "frame_length_bits": response_frame_length,
        "command": (
            f"0x{command:02X}"
            if command is not None
            else None
        ),
        "expected_response": "0xAA",
        "observed_response": (
            f"0x{observed_response:02X}"
            if observed_response is not None
            else None
        ),
        "response_match": response_match,
        "bit_order_error": bit_order_error,
        "miso_stuck_high": miso_stuck_high,
        "miso_stuck_low": miso_stuck_low,
        "clock_glitch": clock_glitch,
        "response_delay_us": response_delay_us,
        "timeout_detected": timeout_detected,
        "miso_transition_count": miso_transition_count,
        "mosi_transition_count": mosi_transition_count
    }


# build the complete feature dataset
def build_feature_dataset():

    raw_log_dir = PROJECT_ROOT / "data" / "raw_logs"

    processed_dir = (
        PROJECT_ROOT /
        "data" /
        "processed"
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path = (
        PROJECT_ROOT /
        "data" /
        "dataset_metadata.csv"
    )

    output_path = (
        processed_dir /
        "spi_features.csv"
    )

    metadata = pd.read_csv(
        metadata_path
    )

    captures = sorted(
        raw_log_dir.glob("capture_*.csv")
    )

    print(
        f"Found {len(captures)} SPI captures."
    )

    print("Extracting features...")

    feature_rows = []

    for index, capture_path in enumerate(
        captures,
        start=1
    ):

        features = extract_features(
            capture_path
        )

        metadata_row = metadata[
            metadata["filename"]
            == capture_path.name
        ]

        if len(metadata_row) == 0:
            raise ValueError(
                f"No metadata found for "
                f"{capture_path.name}"
            )

        failure_type = (
            metadata_row.iloc[0]["failure_type"]
        )

        features["failure_type"] = (
            failure_type
        )

        feature_rows.append(
            features
        )

        if index % 50 == 0:
            print(
                f"Processed "
                f"{index}/{len(captures)} captures"
            )

    dataset = pd.DataFrame(
        feature_rows
    )

    dataset.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"Successfully processed "
        f"{len(dataset)} captures."
    )

    print(
        f"Dataset saved to: "
        f"{output_path}"
    )

    print()
    print(
        f"Dataset shape: "
        f"{dataset.shape}"
    )

    print()
    print("Failure distribution:")

    print(
        dataset["failure_type"]
        .value_counts()
    )

    print()
    print("Feature columns:")

    for column in dataset.columns:
        print(f"- {column}")


# run feature extraction
if __name__ == "__main__":
    build_feature_dataset()