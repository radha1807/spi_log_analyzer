from pathlib import Path

import pandas as pd

from feature_extractor import extract_features
from retriever import retrieve_similar_captures
from llm_analyzer import (
    LLMAnalyzer,
    classify_target_features
)


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

OUTPUT_PATH = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "explanation_quality.csv"
)


# keywords expected in useful evidence for each failure
EVIDENCE_KEYWORDS = {

    "NORMAL": [
    "clock",
    "period"
],

    "MISSING_ACK": [
        "no response",
        "missing",
        "ack",
        "frame",
        "response"
    ],

    "SPI_TIMEOUT": [
        "timeout",
        "timed out",
        "extra",
        "cycles"
    ],

    "WRONG_RESPONSE": [
        "wrong",
        "incorrect",
        "response",
        "expected",
        "observed"
    ],

    "MISO_STUCK_HIGH": [
        "miso",
        "high",
        "stuck",
        "0xff"
    ],

    "MISO_STUCK_LOW": [
        "miso",
        "low",
        "stuck",
        "0x00"
    ],

    "CLOCK_TOO_FAST": [
        "clock",
        "fast",
        "frequency",
        "khz"
    ],

    "CLOCK_TOO_SLOW": [
        "clock",
        "slow",
        "frequency",
        "khz"
    ],

    "BIT_ORDER_ERROR": [
        "bit",
        "order",
        "bit_order"
    ],

    "FRAME_LENGTH_ERROR": [
        "frame",
        "length",
        "bit"
    ],

    "DELAYED_RESPONSE": [
        "delay",
        "delayed",
        "response",
        "us"
    ],

    "GLITCHED_CLOCK": [
        "glitch",
        "clock",
        "period"
    ]
}


# convert values into searchable text
def normalize_text(value):

    if value is None:
        return ""

    return str(value).lower()


# combine evidence statements
def get_evidence_text(diagnosis):

    evidence = diagnosis.get(
        "evidence",
        []
    )

    return " ".join(
        normalize_text(item)
        for item in evidence
    )


# check whether evidence contains relevant terms
def check_evidence_grounding(
    failure_type,
    diagnosis
):

    evidence_text = get_evidence_text(
        diagnosis
    )

    keywords = EVIDENCE_KEYWORDS.get(
        failure_type,
        []
    )

    matches = [
        keyword
        for keyword in keywords
        if keyword in evidence_text
    ]

    return (
        len(matches) > 0,
        matches
    )


# check that the explanation has the required structure
def check_structure(diagnosis):

    required_fields = [
        "failure_type",
        "confidence",
        "evidence",
        "probable_root_cause",
        "recommended_checks"
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in diagnosis
    ]

    if missing_fields:
        return False, (
            f"Missing fields: "
            f"{missing_fields}"
        )

    if not isinstance(
        diagnosis["evidence"],
        list
    ):
        return False, (
            "Evidence is not a list."
        )

    if not isinstance(
        diagnosis["recommended_checks"],
        list
    ):
        return False, (
            "Recommended checks "
            "are not a list."
        )

    if not diagnosis[
        "probable_root_cause"
    ]:
        return False, (
            "Root cause is empty."
        )

    return True, "OK"


# check the expected two recommended checks
def check_recommended_checks(
    diagnosis
):

    checks = diagnosis.get(
        "recommended_checks",
        []
    )

    return len(checks) == 2


# evaluate one explanation
def evaluate_explanation(
    expected_type,
    deterministic_type,
    diagnosis
):

    structure_ok, structure_reason = (
        check_structure(
            diagnosis
        )
    )

    type_ok = (
        diagnosis.get(
            "failure_type"
        ) == expected_type
    )

    evidence_ok, matched_keywords = (
        check_evidence_grounding(
            expected_type,
            diagnosis
        )
    )

    checks_ok = (
        check_recommended_checks(
            diagnosis
        )
    )

    root_cause_ok = bool(
        diagnosis.get(
            "probable_root_cause"
        )
    )

    grounded = all([
        structure_ok,
        type_ok,
        evidence_ok,
        checks_ok,
        root_cause_ok
    ])

    return {
        "structure_ok": structure_ok,
        "type_ok": type_ok,
        "evidence_grounded": evidence_ok,
        "root_cause_present": root_cause_ok,
        "two_checks_present": checks_ok,
        "grounded": grounded,
        "matched_keywords": ", ".join(
            matched_keywords
        ),
        "structure_reason": structure_reason
    }


def main():

    print()
    print("=" * 70)
    print("EXPLANATION QUALITY EVALUATION")
    print("=" * 70)

    metadata = pd.read_csv(
        METADATA_PATH
    )

    representative_rows = (
        metadata
        .sort_values("capture_id")
        .groupby(
            "failure_type",
            sort=True
        )
        .first()
        .reset_index()
    )

    print()
    print(
        f"Testing "
        f"{len(representative_rows)} "
        f"failure classes."
    )

    analyzer = LLMAnalyzer()

    results = []

    for _, row in representative_rows.iterrows():

        expected_type = row[
            "failure_type"
        ]

        filename = row[
            "filename"
        ]

        capture_path = (
            RAW_LOG_DIR /
            filename
        )

        print()
        print("=" * 70)
        print(
            f"Failure class: "
            f"{expected_type}"
        )

        print(
            f"Capture: "
            f"{filename}"
        )

        print("=" * 70)

        try:

            # extract target features
            features = extract_features(
                capture_path
            )

            # determine deterministic diagnosis
            deterministic_type = (
                classify_target_features(
                    features
                )
            )

            # retrieve historical evidence
            (
                _,
                _,
                retrieval_output
            ) = retrieve_similar_captures(
                capture_path,
                top_k=3
            )

            distances = (
                retrieval_output
                .get("distances", [[]])[0]
            )

            metadatas = (
                retrieval_output
                .get("metadatas", [[]])[0]
            )

            documents = (
                retrieval_output
                .get("documents", [[]])[0]
            )

            retrieved_results = []

            for index in range(
                len(distances)
            ):

                retrieved_results.append({
                    "rank": index + 1,
                    "distance": float(
                        distances[index]
                    ),
                    "metadata": (
                        metadatas[index]
                        if index < len(metadatas)
                        else {}
                    ),
                    "document": (
                        documents[index]
                        if index < len(documents)
                        else ""
                    )
                })

            # generate explanation
            diagnosis = analyzer.analyze(
                features,
                retrieved_results
            )

            # evaluate explanation
            evaluation = (
                evaluate_explanation(
                    expected_type,
                    deterministic_type,
                    diagnosis
                )
            )

            print()
            print(
                f"Expected type:       "
                f"{expected_type}"
            )

            print(
                f"Deterministic type:  "
                f"{deterministic_type}"
            )

            print(
                f"LLM type:            "
                f"{diagnosis.get('failure_type')}"
            )

            print(
                f"Evidence grounded:   "
                f"{evaluation['evidence_grounded']}"
            )

            print(
                f"Root cause present:  "
                f"{evaluation['root_cause_present']}"
            )

            print(
                f"Two checks present:  "
                f"{evaluation['two_checks_present']}"
            )

            print(
                f"Overall grounded:    "
                f"{evaluation['grounded']}"
            )

            print()
            print("Evidence:")

            for evidence in diagnosis.get(
                "evidence",
                []
            ):

                print(
                    f"- {evidence}"
                )

            print()
            print("Root cause:")

            print(
                diagnosis.get(
                    "probable_root_cause",
                    ""
                )
            )

            print()
            print("Recommended checks:")

            for check in diagnosis.get(
                "recommended_checks",
                []
            ):

                print(
                    f"- {check}"
                )

            results.append({

                "failure_type": expected_type,

                "file": filename,

                "deterministic_type": (
                    deterministic_type
                ),

                "llm_type": diagnosis.get(
                    "failure_type"
                ),

                "confidence": diagnosis.get(
                    "confidence"
                ),

                "structure_ok": (
                    evaluation[
                        "structure_ok"
                    ]
                ),

                "type_ok": (
                    evaluation[
                        "type_ok"
                    ]
                ),

                "evidence_grounded": (
                    evaluation[
                        "evidence_grounded"
                    ]
                ),

                "root_cause_present": (
                    evaluation[
                        "root_cause_present"
                    ]
                ),

                "two_checks_present": (
                    evaluation[
                        "two_checks_present"
                    ]
                ),

                "grounded": (
                    evaluation[
                        "grounded"
                    ]
                ),

                "matched_keywords": (
                    evaluation[
                        "matched_keywords"
                    ]
                )
            })

        except Exception as error:

            print()
            print(
                f"ERROR: {error}"
            )

            results.append({

                "failure_type": expected_type,

                "file": filename,

                "deterministic_type": None,

                "llm_type": None,

                "confidence": None,

                "structure_ok": False,

                "type_ok": False,

                "evidence_grounded": False,

                "root_cause_present": False,

                "two_checks_present": False,

                "grounded": False,

                "matched_keywords": ""
            })


    results_df = pd.DataFrame(
        results
    )


    # calculate quality metrics
    total = len(
        results_df
    )

    grounded_count = (
        results_df[
            "grounded"
        ].sum()
    )

    evidence_count = (
        results_df[
            "evidence_grounded"
        ].sum()
    )

    structure_count = (
        results_df[
            "structure_ok"
        ].sum()
    )

    checks_count = (
        results_df[
            "two_checks_present"
        ].sum()
    )

    grounded_accuracy = (
        grounded_count / total
        if total > 0
        else 0
    )

    evidence_accuracy = (
        evidence_count / total
        if total > 0
        else 0
    )

    structure_accuracy = (
        structure_count / total
        if total > 0
        else 0
    )

    checks_accuracy = (
        checks_count / total
        if total > 0
        else 0
    )


    print()
    print("=" * 70)
    print("EXPLANATION QUALITY SUMMARY")
    print("=" * 70)

    print()

    print(
        f"Failure classes tested: "
        f"{total}"
    )

    print(
        f"Valid structure: "
        f"{structure_count}/{total} "
        f"({structure_accuracy * 100:.2f}%)"
    )

    print(
        f"Grounded evidence: "
        f"{evidence_count}/{total} "
        f"({evidence_accuracy * 100:.2f}%)"
    )

    print(
        f"Exactly two checks: "
        f"{checks_count}/{total} "
        f"({checks_accuracy * 100:.2f}%)"
    )

    print(
        f"Overall grounded explanations: "
        f"{grounded_count}/{total} "
        f"({grounded_accuracy * 100:.2f}%)"
    )

    print()

    print(
        results_df[
            [
                "failure_type",
                "structure_ok",
                "evidence_grounded",
                "two_checks_present",
                "grounded"
            ]
        ].to_string(
            index=False
        )
    )


    # save evaluation results
    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()

    print(
        f"Results saved to: "
        f"{OUTPUT_PATH}"
    )

    print()


if __name__ == "__main__":
    main()