from pathlib import Path
import sys

import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from feature_extractor import extract_features
from embeddings import create_spi_text
from vector_store import create_chroma_client, create_collection


# define project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_LOG_DIR = PROJECT_ROOT / "data" / "raw_logs"


# define the embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

# keep the model loaded in memory
_embedding_model = None


# load the embedding model once
def load_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        print(
            f"Loading embedding model: {MODEL_NAME}"
        )

        _embedding_model = SentenceTransformer(
            MODEL_NAME
        )

    return _embedding_model


# create an embedding for one spi capture
def create_query_embedding(
    model,
    spi_features
):

    feature_row = pd.Series(
        spi_features
    )

    document = create_spi_text(
        feature_row
    )

    embedding = model.encode(
        document,
        normalize_embeddings=True
    )

    return (
        document,
        embedding.tolist()
    )


# retrieve using already extracted features
def retrieve_similar_from_features(
    features,
    top_k=5
):

    # load the already cached embedding model
    model = load_embedding_model()

    # convert features into semantic text
    document, query_embedding = (
        create_query_embedding(
            model,
            features
        )
    )

    # connect to chromadb
    client = create_chroma_client()

    collection = create_collection(
        client
    )

    # search for similar captures
    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return (
        features,
        document,
        results
    )


# retrieve similar spi captures
def retrieve_similar_captures(
    capture_path,
    top_k=5
):

    capture_path = Path(
        capture_path
    )

    if not capture_path.exists():
        raise FileNotFoundError(
            f"Capture not found: {capture_path}"
        )

    # extract features once
    features = extract_features(
        capture_path
    )

    return retrieve_similar_from_features(
        features,
        top_k=top_k
    )


# display retrieval results
def display_results(
    features,
    document,
    results
):

    print()
    print("Query capture")
    print("-" * 60)

    print(
        f"file: {features['file']}"
    )

    print(
        f"command: {features['command']}"
    )

    print(
        f"observed response: "
        f"{features['observed_response']}"
    )

    print(
        f"clock frequency: "
        f"{features['clock_frequency_khz']} kHz"
    )

    print(
        f"frame length: "
        f"{features['frame_length_bits']} bits"
    )

    print(
        f"response delay: "
        f"{features['response_delay_us']} us"
    )

    print()
    print("Retrieved similar captures")
    print("-" * 60)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for index in range(len(ids)):

        similarity = 1 - distances[index]

        metadata = metadatas[index]

        print()
        print(
            f"Result {index + 1}"
        )

        print(
            f"capture: {metadata['file']}"
        )

        print(
            f"similarity: {similarity:.4f}"
        )

        print(
            f"command: {metadata['command']}"
        )

        print(
            f"observed response: "
            f"{metadata['observed_response']}"
        )

        print(
            f"clock frequency: "
            f"{metadata['clock_frequency_khz']:.2f} kHz"
        )

        print(
            f"frame length: "
            f"{metadata['frame_length_bits']} bits"
        )

        print(
            f"response match: "
            f"{metadata['response_match']}"
        )

        print(
            f"bit order error: "
            f"{metadata['bit_order_error']}"
        )

        print(
            f"MISO stuck high: "
            f"{metadata['miso_stuck_high']}"
        )

        print(
            f"MISO stuck low: "
            f"{metadata['miso_stuck_low']}"
        )

        print(
            f"clock glitch: "
            f"{metadata['clock_glitch']}"
        )

        print(
            f"timeout: "
            f"{metadata['timeout_detected']}"
        )


# run a retrieval test
if __name__ == "__main__":

    capture_path = (
        RAW_LOG_DIR /
        "capture_0001.csv"
    )

    features, document, results = (
        retrieve_similar_captures(
            capture_path,
            top_k=5
        )
    )

    display_results(
        features,
        document,
        results
    )