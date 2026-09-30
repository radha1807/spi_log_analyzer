from pathlib import Path
import random
import pandas as pd


# define project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_LOG_DIR = PROJECT_ROOT / "data" / "raw_logs"
DATA_DIR = PROJECT_ROOT / "data"

RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# convert a byte into lsb-first bits
def byte_to_bits(value):
    return [
        (value >> index) & 1
        for index in range(8)
    ]


# add one clock cycle to the waveform
def add_clock_cycle(
    rows,
    timestamp,
    period_us,
    mosi_bit,
    miso_bit
):
    half_period = period_us / 2

    rows.append({
        "timestamp_us": round(timestamp, 3),
        "clk": 1,
        "mosi": mosi_bit,
        "miso": miso_bit
    })

    timestamp += half_period

    rows.append({
        "timestamp_us": round(timestamp, 3),
        "clk": 0,
        "mosi": mosi_bit,
        "miso": miso_bit
    })

    timestamp += half_period

    return timestamp


# create response bits
def create_response_bits(observed_response):

    if observed_response == "0xAA":
        return byte_to_bits(0xAA)

    if observed_response == "0x55":
        return byte_to_bits(0x55)

    if observed_response == "0x33":
        return byte_to_bits(0x33)

    if observed_response == "0xFF":
        return byte_to_bits(0xFF)

    return []


# generate one spi capture
def generate_capture(capture_id, failure_type):

    command = random.choice([
        "0x05",
        "0x06",
        "0x0A"
    ])

    expected_response = "0xAA"
    observed_response = "0xAA"

    clock_frequency_khz = random.choice([
        77,
        100,
        250
    ])

    response_delay_us = 0
    frame_length = 8
    clock_glitch = 0

    # set failure-specific clock parameters
    if failure_type == "CLOCK_TOO_FAST":

        clock_frequency_khz = random.choice([
            500,
            750,
            1000
        ])

    elif failure_type == "CLOCK_TOO_SLOW":

        clock_frequency_khz = random.choice([
            5,
            10,
            20
        ])

    # define response behavior
    if failure_type == "MISSING_ACK":

        observed_response = None
        frame_length = 0

    elif failure_type == "WRONG_RESPONSE":

        observed_response = "0x33"

    elif failure_type == "BIT_ORDER_ERROR":

        observed_response = "0x55"

    elif failure_type == "DELAYED_RESPONSE":

        response_delay_us = random.choice([
            30,
            50,
            80
        ])

    elif failure_type == "SPI_TIMEOUT":

        observed_response = None
        frame_length = 0

    elif failure_type == "FRAME_LENGTH_ERROR":

        frame_length = random.choice([
            6,
            7,
            9,
            10
        ])

    elif failure_type == "GLITCHED_CLOCK":

        clock_glitch = 1

    clock_period_us = (
        1000 /
        clock_frequency_khz
    )

    # create command bits
    command_bits = byte_to_bits(
        int(command, 16)
    )

    # create response bits
    response_bits = create_response_bits(
        observed_response
    )

    rows = []
    timestamp = 0.0

    # generate command frame
    for bit in command_bits:

        timestamp = add_clock_cycle(
            rows,
            timestamp,
            clock_period_us,
            bit,
            1
        )

    # simulate missing ack
    if failure_type == "MISSING_ACK":

        pass

    # simulate timeout
    elif failure_type == "SPI_TIMEOUT":

        for _ in range(12):

            timestamp = add_clock_cycle(
                rows,
                timestamp,
                clock_period_us,
                0,
                1
            )

    # simulate delayed response
    elif failure_type == "DELAYED_RESPONSE":

        timestamp += response_delay_us

        for bit in response_bits:

            timestamp = add_clock_cycle(
                rows,
                timestamp,
                clock_period_us,
                0,
                bit
            )

    else:

        # generate response frame
        if failure_type == "FRAME_LENGTH_ERROR":

            response_bits_to_send = []

            for index in range(frame_length):

                response_bits_to_send.append(
                    response_bits[
                        index % len(response_bits)
                    ]
                )

        else:

            response_bits_to_send = response_bits[:8]

        for index, bit in enumerate(
            response_bits_to_send
        ):

            miso_bit = bit

            # simulate MISO stuck high
            if failure_type == "MISO_STUCK_HIGH":

                miso_bit = 1

            # simulate MISO stuck low
            elif failure_type == "MISO_STUCK_LOW":

                miso_bit = 0

            period = clock_period_us

            # simulate clock glitch
            if (
                failure_type == "GLITCHED_CLOCK"
                and index == 3
            ):

                period = (
                    clock_period_us *
                    0.25
                )

            timestamp = add_clock_cycle(
                rows,
                timestamp,
                period,
                0,
                miso_bit
            )

    # save waveform
    filename = (
        f"capture_{capture_id:04d}.csv"
    )

    output_path = (
        RAW_LOG_DIR /
        filename
    )

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        output_path,
        index=False
    )

    return {
        "capture_id": capture_id,
        "filename": filename,
        "failure_type": failure_type,
        "command": command,
        "expected_response": expected_response,
        "observed_response": observed_response,
        "clock_frequency_khz": clock_frequency_khz,
        "response_delay_us": response_delay_us,
        "frame_length": frame_length,
        "clock_glitch": clock_glitch
    }


# define available failure classes
failure_types = [
    "NORMAL",
    "MISSING_ACK",
    "SPI_TIMEOUT",
    "WRONG_RESPONSE",
    "MISO_STUCK_HIGH",
    "MISO_STUCK_LOW",
    "CLOCK_TOO_FAST",
    "CLOCK_TOO_SLOW",
    "BIT_ORDER_ERROR",
    "FRAME_LENGTH_ERROR",
    "DELAYED_RESPONSE",
    "GLITCHED_CLOCK"
]


# generate the complete synthetic dataset
def generate_dataset(total_captures=500):

    metadata = []

    for capture_id in range(
        1,
        total_captures + 1
    ):

        failure_type = random.choice(
            failure_types
        )

        capture_metadata = generate_capture(
            capture_id,
            failure_type
        )

        metadata.append(
            capture_metadata
        )

        if capture_id % 50 == 0:

            print(
                f"Generated "
                f"{capture_id}/{total_captures} "
                f"captures"
            )

    metadata_df = pd.DataFrame(
        metadata
    )

    metadata_path = (
        DATA_DIR /
        "dataset_metadata.csv"
    )

    metadata_df.to_csv(
        metadata_path,
        index=False
    )

    print()

    print(
        f"Generated "
        f"{total_captures} SPI captures."
    )

    print(
        f"Raw logs saved to: "
        f"{RAW_LOG_DIR}"
    )

    print(
        f"Metadata saved to: "
        f"{metadata_path}"
    )


# run dataset generation
if __name__ == "__main__":

    generate_dataset(500)