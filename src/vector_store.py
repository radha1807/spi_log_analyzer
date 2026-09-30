from pathlib import Path

import chromadb

from embeddings import build_embeddings


# define project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"


# create chroma client
def create_chroma_client():
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


# create or load the spi collection
def create_collection(client):
    return client.get_or_create_collection(
        name="spi_failures"
    )


# store spi embeddings in chromadb
def build_vector_store():
    df, documents, embeddings = build_embeddings()

    client = create_chroma_client()

    collection = create_collection(client)

    ids = [
        f"capture_{index + 1:04d}"
        for index in range(len(df))
    ]

    # store metadata without the ground-truth label
    metadatas = []

    for _, row in df.iterrows():
        metadata = {
            "file": str(row["file"]),
            "command": str(row["command"]),
            "expected_response": str(
                row["expected_response"]
            ),
            "observed_response": str(
                row["observed_response"]
            ),
            "clock_frequency_khz": float(
                row["clock_frequency_khz"]
            ),
            "frame_length_bits": int(
                row["frame_length_bits"]
            ),
            "response_match": int(
                row["response_match"]
            ),
            "bit_order_error": int(
                row["bit_order_error"]
            ),
            "miso_stuck_high": int(
                row["miso_stuck_high"]
            ),
            "miso_stuck_low": int(
                row["miso_stuck_low"]
            ),
            "clock_glitch": int(
                row["clock_glitch"]
            ),
            "timeout_detected": int(
                row["timeout_detected"]
            )
        }

        metadatas.append(metadata)

    print("Adding embeddings to ChromaDB...")

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Stored {collection.count()} captures "
        "in ChromaDB."
    )

    print(
        f"ChromaDB location: {CHROMA_DIR}"
    )


# retrieve similar spi captures
def search_similar_captures(
    query_embedding,
    top_k=5
):
    client = create_chroma_client()

    collection = create_collection(client)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


# build the vector database
if __name__ == "__main__":
    build_vector_store()