import json
import os

from google import genai
from google.genai import types


# define the default provider
DEFAULT_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "ollama"
).lower()


# define the gemini model
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


# define the ollama model
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b"
)


# define allowed failure categories
FAILURE_CATEGORIES = [
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


# define the expected analysis response
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "failure_type": {
            "type": "string"
        },
        "confidence": {
            "type": "number"
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "probable_root_cause": {
            "type": "string"
        },
        "recommended_checks": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "failure_type",
        "confidence",
        "evidence",
        "probable_root_cause",
        "recommended_checks"
    ]
}


# convert a value to a boolean
def is_true(value):
    return str(value).lower() in {
        "1",
        "true",
        "yes"
    }


# safely convert a value to float
def safe_float(value, default=0.0):

    try:
        if value is None:
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


# determine failure type from measured target features
def classify_target_features(features):

    frame_length = safe_float(
        features.get("frame_length_bits"),
        0
    )

    response_delay = safe_float(
        features.get("response_delay_us"),
        0
    )

    clock_frequency = safe_float(
        features.get("clock_frequency_khz"),
        0
    )

    expected_response = (
        features.get("expected_response")
    )

    observed_response = (
        features.get("observed_response")
    )

    response_match = is_true(
        features.get("response_match")
    )

    bit_order_error = is_true(
        features.get("bit_order_error")
    )

    miso_stuck_high = is_true(
        features.get("miso_stuck_high")
    )

    miso_stuck_low = is_true(
        features.get("miso_stuck_low")
    )

    clock_glitch = is_true(
        features.get("clock_glitch")
    )

    timeout_detected = is_true(
        features.get("timeout_detected")
    )

    # timeout has priority over generic frame-length errors
    # timeout has priority
    if timeout_detected:
        return "SPI_TIMEOUT"

# dedicated clock failures
    if clock_frequency >= 500:
        return "CLOCK_TOO_FAST"

    if clock_frequency <= 20:
        return "CLOCK_TOO_SLOW"

    # no response frame
    if frame_length == 0:
        return "MISSING_ACK"

    # stuck MISO
    if miso_stuck_high:
        return "MISO_STUCK_HIGH"

    if miso_stuck_low:
        return "MISO_STUCK_LOW"

    # clock glitch
    if clock_glitch:
        return "GLITCHED_CLOCK"

    # delayed response
    if response_delay > 1:
        return "DELAYED_RESPONSE"

    # bit ordering problem
    if bit_order_error:
        return "BIT_ORDER_ERROR"

    # invalid response frame size
    if frame_length != 8:
        return "FRAME_LENGTH_ERROR"

    # valid frame but incorrect response
    if (
        observed_response is not None
        and expected_response is not None
        and not response_match
    ):
        return "WRONG_RESPONSE"

    return "NORMAL"

# calculate deterministic confidence
def calculate_target_confidence(
    features,
    failure_type
):

    frame_length = safe_float(
        features.get("frame_length_bits"),
        0
    )

    response_delay = safe_float(
        features.get("response_delay_us"),
        0
    )

    if failure_type == "MISSING_ACK":
        return 1.0 if frame_length == 0 else 0.0

    if failure_type == "SPI_TIMEOUT":
        return 1.0 if is_true(
            features.get("timeout_detected")
        ) else 0.0

    if failure_type == "MISO_STUCK_HIGH":
        return 1.0 if is_true(
            features.get("miso_stuck_high")
        ) else 0.0

    if failure_type == "MISO_STUCK_LOW":
        return 1.0 if is_true(
            features.get("miso_stuck_low")
        ) else 0.0

    if failure_type == "GLITCHED_CLOCK":
        return 1.0 if is_true(
            features.get("clock_glitch")
        ) else 0.0

    if failure_type == "DELAYED_RESPONSE":
        return 1.0 if response_delay > 1 else 0.0

    if failure_type == "BIT_ORDER_ERROR":
        return 1.0 if is_true(
            features.get("bit_order_error")
        ) else 0.0

    if failure_type == "FRAME_LENGTH_ERROR":
        return 1.0 if frame_length != 8 else 0.0

    if failure_type == "WRONG_RESPONSE":
        return 1.0 if (
            frame_length == 8
            and not is_true(
                features.get("response_match")
            )
        ) else 0.0

    return 1.0


# build the prompt sent to the llm
def build_analysis_prompt(
    query_features,
    retrieved_results,
    target_failure_type
):

    evidence_lines = []

    # use only the strongest retrieved examples
    for result in retrieved_results[:3]:

        metadata = result.get(
            "metadata",
            {}
        )

        evidence_lines.append(
            f"Historical example {result.get('rank')}: "
            f"failure={metadata.get('failure_type', 'unknown')}, "
            f"command={metadata.get('command')}, "
            f"response={metadata.get('observed_response')}, "
            f"clock={metadata.get('clock_frequency_khz')} kHz, "
            f"frame={metadata.get('frame_length_bits')}, "
            f"response_match={metadata.get('response_match')}, "
            f"bit_order_error={metadata.get('bit_order_error')}, "
            f"miso_high={metadata.get('miso_stuck_high')}, "
            f"miso_low={metadata.get('miso_stuck_low')}, "
            f"clock_glitch={metadata.get('clock_glitch')}, "
            f"timeout={metadata.get('timeout_detected')}"
        )

    retrieved_evidence = "\n".join(
        evidence_lines
    )

    return f"""
You are an SPI debugging assistant.

The deterministic classifier has already identified the target failure.

Target failure:
{target_failure_type}

You MUST keep this exact failure_type.
Do not classify the target again.

Use only the target measurements below as evidence for the target.
Historical captures are supporting context only.
Never copy measurements from historical captures into the target.

Target measurements:
clock={query_features.get('clock_frequency_khz')} kHz
period={query_features.get('clock_period_us')} us
frame_length={query_features.get('frame_length_bits')}
command={query_features.get('command')}
expected_response={query_features.get('expected_response')}
observed_response={query_features.get('observed_response')}
response_match={query_features.get('response_match')}
bit_order_error={query_features.get('bit_order_error')}
miso_stuck_high={query_features.get('miso_stuck_high')}
miso_stuck_low={query_features.get('miso_stuck_low')}
clock_glitch={query_features.get('clock_glitch')}
response_delay={query_features.get('response_delay_us')} us
timeout={query_features.get('timeout_detected')}
miso_transitions={query_features.get('miso_transition_count')}
mosi_transitions={query_features.get('mosi_transition_count')}

Historical context:
{retrieved_evidence}

Return ONLY JSON.

The JSON must contain exactly:

{{
    "failure_type": "{target_failure_type}",
    "confidence": 1.0,
    "evidence": [
        "target measurement supporting diagnosis",
        "target measurement supporting diagnosis"
    ],
    "probable_root_cause": "specific technical explanation based only on target evidence",
    "recommended_checks": [
        "specific hardware or software check",
        "specific hardware or software check"
    ]
}}

Rules:
- failure_type must be exactly "{target_failure_type}"
- confidence must be between 0 and 1
- evidence must refer to target measurements
- do not invent measurements
- do not use historical measurements as target measurements
- keep evidence concise
- keep root cause concise
- provide exactly 2 recommended checks
"""

# validate the analysis structure
def validate_analysis(result):

    required_fields = [
        "failure_type",
        "confidence",
        "evidence",
        "probable_root_cause",
        "recommended_checks"
    ]

    for field in required_fields:

        if field not in result:

            raise ValueError(
                f"Missing required analysis field: {field}"
            )

    confidence = result["confidence"]

    if not isinstance(
        confidence,
        (int, float)
    ):

        raise ValueError(
            "confidence must be a number."
        )

    if not 0 <= confidence <= 1:

        raise ValueError(
            "confidence must be between 0 and 1."
        )

    if not isinstance(
        result["failure_type"],
        str
    ):

        raise ValueError(
            "failure_type must be a string."
        )

    if result["failure_type"] not in FAILURE_CATEGORIES:

        raise ValueError(
            "Unknown failure category: "
            f"{result['failure_type']}"
        )

    if not isinstance(
        result["evidence"],
        list
    ):

        raise ValueError(
            "evidence must be a list."
        )

    if not isinstance(
        result["probable_root_cause"],
        str
    ):

        raise ValueError(
            "probable_root_cause must be a string."
        )

    if not isinstance(
        result["recommended_checks"],
        list
    ):

        raise ValueError(
            "recommended_checks must be a list."
        )


# parse llm response
def parse_llm_response(response):

    if isinstance(
        response,
        dict
    ):

        result = response

    else:

        response = response.strip()

        if response.startswith("```"):

            response = response.replace(
                "```json",
                ""
            )

            response = response.replace(
                "```",
                ""
            )

            response = response.strip()

        try:

            result = json.loads(
                response
            )

        except json.JSONDecodeError as error:

            raise ValueError(
                f"LLM returned invalid JSON: {error}"
            ) from error

    validate_analysis(
        result
    )

    return result


# create the gemini provider
class GeminiProvider:

    def __init__(
        self,
        model_name=GEMINI_MODEL
    ):

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:

            raise RuntimeError(
                "GEMINI_API_KEY environment "
                "variable is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model_name = model_name

    # generate a response with gemini
    def generate(
        self,
        prompt
    ):

        try:

            response = (
                self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                        response_schema=ANALYSIS_SCHEMA
                    )
                )
            )

            if not response.text:

                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text

        except Exception as error:

            raise RuntimeError(
                f"Gemini request failed: {error}"
            ) from error


# create the ollama provider
class OllamaProvider:

    def __init__(
        self,
        model_name=OLLAMA_MODEL
    ):

        self.model_name = model_name

    # generate a response with ollama
    def generate(
        self,
        prompt
    ):

        try:

            import requests

        except ImportError as error:

            raise RuntimeError(
                "The requests package is required "
                "for Ollama."
            ) from error

        url = (
            "http://localhost:11434/api/generate"
        )

        payload = {
    "model": self.model_name,
    "prompt": prompt,
    "stream": False,
    "format": "json",
    "keep_alive": "10m",
    "options": {
        "temperature": 0.1,
        "num_predict": 180,
        "num_ctx": 2048
    }
}
        
        try:

            response = requests.post(
                url,
                json=payload,
                timeout=120
            )

        except requests.exceptions.ConnectionError as error:

            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            ) from error

        except requests.exceptions.Timeout as error:

            raise RuntimeError(
                "Ollama request timed out."
            ) from error

        if response.status_code != 200:

            raise RuntimeError(
                "Ollama returned HTTP "
                f"{response.status_code}: "
                f"{response.text}"
            )

        data = response.json()

        response_text = data.get(
            "response"
        )

        if not response_text:

            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return response_text


# create the llm analyzer
class LLMAnalyzer:

    def __init__(
        self,
        provider=None
    ):

        if provider is not None:

            self.provider = provider

            return

        if DEFAULT_PROVIDER == "gemini":

            self.provider = GeminiProvider()

        elif DEFAULT_PROVIDER == "ollama":

            self.provider = OllamaProvider()

        else:

            raise ValueError(
                "Unsupported LLM provider: "
                f"{DEFAULT_PROVIDER}"
            )

    # generate the final diagnosis
    def analyze(
        self,
        query_features,
        retrieved_results
    ):

        target_failure_type = (
            classify_target_features(
                query_features
            )
        )

        target_confidence = (
            calculate_target_confidence(
                query_features,
                target_failure_type
            )
        )

        prompt = build_analysis_prompt(
            query_features,
            retrieved_results,
            target_failure_type
        )

        response = self.provider.generate(
            prompt
        )

        diagnosis = parse_llm_response(
            response
        )

        # enforce deterministic classification
        diagnosis["failure_type"] = (
            target_failure_type
        )

        # prevent llm from returning zero
        # when the measured target features
        # provide a deterministic diagnosis
        if target_confidence > 0:
            diagnosis["confidence"] = (
                target_confidence
            )

        return diagnosis


# test the selected provider
if __name__ == "__main__":

    print(
        f"LLM provider: {DEFAULT_PROVIDER}"
    )

    if DEFAULT_PROVIDER == "ollama":

        print(
            f"Ollama model: {OLLAMA_MODEL}"
        )

    elif DEFAULT_PROVIDER == "gemini":

        print(
            f"Gemini model: {GEMINI_MODEL}"
        )

    analyzer = LLMAnalyzer()

    test_features = {
        "clock_frequency_khz": 100.0,
        "clock_period_us": 10.0,
        "frame_length_bits": 0,
        "command": "0x0A",
        "expected_response": "0xAA",
        "observed_response": None,
        "response_match": 0,
        "bit_order_error": 0,
        "miso_stuck_high": 0,
        "miso_stuck_low": 0,
        "clock_glitch": 0,
        "response_delay_us": None,
        "timeout_detected": 0,
        "miso_transition_count": 0,
        "mosi_transition_count": 3
    }

    test_results = [
        {
            "rank": 1,
            "distance": 0.0052,
            "metadata": {
                "command": "0x0A",
                "expected_response": "0xAA",
                "observed_response": "0xFF",
                "clock_frequency_khz": 100.0,
                "frame_length_bits": 8,
                "response_match": 0,
                "bit_order_error": 0,
                "miso_stuck_high": 1,
                "miso_stuck_low": 0,
                "clock_glitch": 0,
                "timeout_detected": 0
            }
        }
    ]

    diagnosis = analyzer.analyze(
        test_features,
        test_results
    )

    print("\nLLM diagnosis:")

    print(
        json.dumps(
            diagnosis,
            indent=4
        )
    )