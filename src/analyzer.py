from pathlib import Path
import sys
import pandas as pd

# add src directory to python path
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_extractor import extract_features
from retriever import retrieve_similar_captures
from llm_analyzer import LLMAnalyzer


# define project directories
PROJECT_ROOT = SRC_DIR.parent
RAW_LOG_DIR = PROJECT_ROOT / "data" / "raw_logs"


# prepare retrieved results for the llm
def prepare_retrieval_results(retrieval_output):
    distances = retrieval_output.get("distances", [[]])[0]
    metadatas = retrieval_output.get("metadatas", [[]])[0]
    documents = retrieval_output.get("documents", [[]])[0]

    results = []

    for index, (distance, metadata, document) in enumerate(
        zip(distances, metadatas, documents)
    ):
        results.append({
            "rank": index + 1,
            "distance": float(distance),
            "metadata": metadata,
            "document": document
        })

    return results


# display retrieved evidence
def display_retrieved_evidence(retrieved_results):

    print("\nRetrieved evidence:")

    for result in retrieved_results:
        metadata = result["metadata"]

        print(
            f"\nResult {result['rank']}: "
            f"distance={result['distance']:.4f}"
        )

        print(
            f"  command: "
            f"{metadata.get('command')}"
        )

        print(
            f"  clock frequency: "
            f"{metadata.get('clock_frequency_khz')} kHz"
        )

        print(
            f"  frame length: "
            f"{metadata.get('frame_length_bits')} bits"
        )

        print(
            f"  response match: "
            f"{metadata.get('response_match')}"
        )

        print(
            f"  bit order error: "
            f"{metadata.get('bit_order_error')}"
        )

        print(
            f"  MISO stuck high: "
            f"{metadata.get('miso_stuck_high')}"
        )

        print(
            f"  MISO stuck low: "
            f"{metadata.get('miso_stuck_low')}"
        )

        print(
            f"  clock glitch: "
            f"{metadata.get('clock_glitch')}"
        )

        print(
            f"  timeout: "
            f"{metadata.get('timeout_detected')}"
        )


# analyze one spi capture
def analyze_capture(capture_path, top_k=5, llm_provider=None):

    capture_path = Path(capture_path)

    if not capture_path.exists():
        raise FileNotFoundError(
            f"Capture file not found: {capture_path}"
        )

    print("\nAnalyzing SPI capture:")
    print(capture_path.name)

    # extract spi features
    features = extract_features(capture_path)

    feature_series = pd.Series(features)

    print("\nExtracted features:")

    print(
        f"clock frequency: "
        f"{feature_series.get('clock_frequency_khz')} kHz"
    )

    print(
        f"frame length: "
        f"{feature_series.get('frame_length_bits')} bits"
    )

    print(
        f"command: "
        f"{feature_series.get('command')}"
    )

    print(
        f"observed response: "
        f"{feature_series.get('observed_response')}"
    )

    # retrieve similar captures
    print("\nSearching historical SPI captures...")

    retrieval_features, retrieval_document, retrieval_output = (
        retrieve_similar_captures(
            capture_path,
            top_k=top_k
        )
    )

    retrieved_results = prepare_retrieval_results(
        retrieval_output
    )

    print(
        f"Retrieved {len(retrieved_results)} "
        f"similar captures."
    )

    display_retrieved_evidence(
        retrieved_results
    )

    # create llm analyzer
    analyzer = LLMAnalyzer(
        provider=llm_provider
    )

    # generate diagnosis
    diagnosis = analyzer.analyze(
        features,
        retrieved_results
    )

    return {
        "capture": capture_path.name,
        "features": features,
        "retrieved_results": retrieved_results,
        "diagnosis": diagnosis
    }


# print final diagnosis
def display_diagnosis(result):

    diagnosis = result["diagnosis"]

    print("\n")
    print("=" * 60)
    print("SPI DIAGNOSIS")
    print("=" * 60)

    print(
        f"\nFailure type: "
        f"{diagnosis['failure_type']}"
    )

    print(
        f"Confidence: "
        f"{diagnosis['confidence']:.2f}"
    )

    print("\nEvidence:")

    for item in diagnosis["evidence"]:
        print(f"- {item}")

    print("\nProbable root cause:")

    print(
        diagnosis["probable_root_cause"]
    )

    print("\nRecommended checks:")

    for item in diagnosis["recommended_checks"]:
        print(f"- {item}")

    print("\n" + "=" * 60)


# run the analyzer
if __name__ == "__main__":

    test_capture = (
        RAW_LOG_DIR / "capture_0001.csv"
    )

    try:

        result = analyze_capture(
            test_capture,
            top_k=5
        )

        display_diagnosis(result)

    except Exception as error:

        print(
            f"\nAnalysis failed: {error}"
        )