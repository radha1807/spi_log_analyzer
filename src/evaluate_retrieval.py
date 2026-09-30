from pathlib import Path
import sys
import time

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

sys.path.append(
    str(Path(__file__).resolve().parent)
)

from embeddings import create_spi_text
from vector_store import create_chroma_client


# define project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spi_features.csv"
)


# define embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


# define evaluation collection
COLLECTION_NAME = "spi_failures_evaluation"


# load the embedding model
def load_embedding_model():
    print(
        f"Loading embedding model: {MODEL_NAME}"
    )

    return SentenceTransformer(
        MODEL_NAME
    )


# create train and test datasets
def create_train_test_split(df):
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["failure_type"]
    )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


# create the evaluation collection
def create_evaluation_collection(client):
    existing_collections = [
        collection.name
        for collection in client.list_collections()
    ]

    if COLLECTION_NAME in existing_collections:
        client.delete_collection(
            COLLECTION_NAME
        )

    return client.create_collection(
        name=COLLECTION_NAME
    )


# build the vector database using only training data
def build_training_store(
    collection,
    model,
    train_df
):
    documents = []

    for _, row in train_df.iterrows():
        document = create_spi_text(row)

        documents.append(document)

    print(
        f"Creating embeddings for "
        f"{len(documents)} training captures..."
    )

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    ids = [
        f"train_{index:04d}"
        for index in range(len(train_df))
    ]

    metadatas = []

    for _, row in train_df.iterrows():
        metadata = {
            "file": str(row["file"]),
            "failure_type": str(
                row["failure_type"]
            )
        }

        metadatas.append(metadata)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(
        f"Training collection contains "
        f"{collection.count()} captures."
    )


# retrieve similar failures for one test capture
def retrieve_for_test_capture(
    collection,
    model,
    row,
    top_k=5
):
    document = create_spi_text(row)

    query_embedding = model.encode(
        document,
        normalize_embeddings=True
    ).tolist()

    start_time = time.perf_counter()

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k,
        include=[
            "metadatas",
            "distances"
        ]
    )

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return results, elapsed_ms


# calculate retrieval accuracy
def evaluate_retrieval(
    collection,
    model,
    test_df
):
    top_1_correct = 0
    top_3_correct = 0
    top_5_correct = 0

    latencies = []

    results_table = []

    for index, (_, row) in enumerate(
        test_df.iterrows(),
        start=1
    ):
        true_failure = str(
            row["failure_type"]
        )

        results, elapsed_ms = (
            retrieve_for_test_capture(
                collection,
                model,
                row,
                top_k=5
            )
        )

        latencies.append(
            elapsed_ms
        )

        retrieved_metadata = (
            results["metadatas"][0]
        )

        retrieved_failures = [
            metadata["failure_type"]
            for metadata in retrieved_metadata
        ]

        top_1_failure = (
            retrieved_failures[0]
            if len(retrieved_failures) >= 1
            else None
        )

        top_3_failures = retrieved_failures[:3]
        top_5_failures = retrieved_failures[:5]

        top_1_hit = (
            true_failure == top_1_failure
        )

        top_3_hit = (
            true_failure in top_3_failures
        )

        top_5_hit = (
            true_failure in top_5_failures
        )

        if top_1_hit:
            top_1_correct += 1

        if top_3_hit:
            top_3_correct += 1

        if top_5_hit:
            top_5_correct += 1

        results_table.append({
            "file": row["file"],
            "true_failure": true_failure,
            "top_1_failure": top_1_failure,
            "top_1_hit": int(top_1_hit),
            "top_3_hit": int(top_3_hit),
            "top_5_hit": int(top_5_hit),
            "latency_ms": elapsed_ms
        })

        if index % 10 == 0:
            print(
                f"Evaluated "
                f"{index}/{len(test_df)} captures"
            )

    total = len(test_df)

    top_1_accuracy = (
        top_1_correct / total
    )

    top_3_accuracy = (
        top_3_correct / total
    )

    top_5_accuracy = (
        top_5_correct / total
    )

    average_latency = (
        sum(latencies) / len(latencies)
    )

    return {
        "top_1_accuracy": top_1_accuracy,
        "top_3_accuracy": top_3_accuracy,
        "top_5_accuracy": top_5_accuracy,
        "average_latency_ms": average_latency,
        "results": results_table
    }


# display evaluation results
def display_results(
    train_df,
    test_df,
    evaluation
):
    print()
    print("=" * 60)
    print("SPI RETRIEVAL EVALUATION")
    print("=" * 60)

    print()
    print(
        f"Total captures: "
        f"{len(train_df) + len(test_df)}"
    )

    print(
        f"Training captures: "
        f"{len(train_df)}"
    )

    print(
        f"Test captures: "
        f"{len(test_df)}"
    )

    print()
    print(
        f"Top-1 accuracy: "
        f"{evaluation['top_1_accuracy']:.2%}"
    )

    print(
        f"Top-3 accuracy: "
        f"{evaluation['top_3_accuracy']:.2%}"
    )

    print(
        f"Top-5 accuracy: "
        f"{evaluation['top_5_accuracy']:.2%}"
    )

    print(
        f"Average retrieval latency: "
        f"{evaluation['average_latency_ms']:.2f} ms"
    )


# run retrieval evaluation
def main():
    print(
        f"Loading dataset from: "
        f"{DATASET_PATH}"
    )

    df = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"Loaded {len(df)} captures."
    )

    train_df, test_df = (
        create_train_test_split(df)
    )

    print()
    print(
        f"Training set: {len(train_df)}"
    )

    print(
        f"Test set: {len(test_df)}"
    )

    model = load_embedding_model()

    client = create_chroma_client()

    collection = (
        create_evaluation_collection(
            client
        )
    )

    build_training_store(
        collection,
        model,
        train_df
    )

    evaluation = evaluate_retrieval(
        collection,
        model,
        test_df
    )

    display_results(
        train_df,
        test_df,
        evaluation
    )

    results_df = pd.DataFrame(
        evaluation["results"]
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "retrieval_evaluation.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"Detailed results saved to: "
        f"{output_path}"
    )


# start evaluation
if __name__ == "__main__":
    main()