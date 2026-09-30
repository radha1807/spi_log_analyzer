from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer


# define project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "spi_features.csv"


# define the local embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


# load the embedding model
def load_embedding_model():
    print(
        f"Loading embedding model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Embedding model loaded.")

    return model


# convert a boolean feature into readable text
def yes_no(value):
    return "yes" if int(value) == 1 else "no"


# convert one spi capture into semantic text
def create_spi_text(row):
    response_delay = row["response_delay_us"]

    if pd.isna(response_delay):
        response_delay_text = "not observed"
    else:
        response_delay_text = (
            f"{response_delay:.2f} us"
        )

    text = f"""
SPI capture characteristics:

clock frequency: {row["clock_frequency_khz"]:.2f} kHz
clock period: {row["clock_period_us"]:.2f} us
frame length: {int(row["frame_length_bits"])} bits

command sent: {row["command"]}
expected response: {row["expected_response"]}
observed response: {row["observed_response"]}

response matched expected value: {yes_no(row["response_match"])}
bit order anomaly detected: {yes_no(row["bit_order_error"])}

MISO stuck high: {yes_no(row["miso_stuck_high"])}
MISO stuck low: {yes_no(row["miso_stuck_low"])}

clock anomaly detected: {yes_no(row["clock_glitch"])}

response delay: {response_delay_text}
timeout detected: {yes_no(row["timeout_detected"])}

MISO transition count: {int(row["miso_transition_count"])}
MOSI transition count: {int(row["mosi_transition_count"])}
"""

    return " ".join(
        text.split()
    )


# generate local embeddings
def generate_embeddings(
    texts,
    model
):
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings.tolist()


# load the processed spi dataset
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    return pd.read_csv(
        DATASET_PATH
    )


# build embeddings for the complete dataset
def build_embeddings():
    df = load_dataset()

    print(
        f"Loaded {len(df)} SPI captures."
    )

    # create semantic representation for every capture
    documents = [
        create_spi_text(row)
        for _, row in df.iterrows()
    ]

    print(
        "Created SPI text representations."
    )

    # load the local embedding model
    model = load_embedding_model()

    # generate embeddings
    print(
        "Generating local embeddings..."
    )

    embeddings = generate_embeddings(
        documents,
        model
    )

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    return df, documents, embeddings


# run embedding generation as a script
if __name__ == "__main__":
    df, documents, embeddings = build_embeddings()

    print()
    print("First SPI representation:")
    print(documents[0])

    print()
    print("Embedding dimensions:")
    print(
        len(embeddings[0])
    )