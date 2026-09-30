import unittest
from pathlib import Path
from src.parser import parse_capture

class ParserTests(unittest.TestCase):
    def test_capture_has_expected_fields(self):
        path = Path("data/raw_logs/capture_0001.csv")
        if not path.exists():
            self.skipTest("Run: python -m src.generate_dataset")
        result = parse_capture(path)
        for key in [
            "capture_id", "command", "expected_response",
            "observed_response", "bit_count", "duration_us",
            "parser_label"
        ]:
            self.assertIn(key, result)

    def test_bit_count_is_numeric(self):
        path = Path("data/raw_logs/capture_0001.csv")
        if not path.exists():
            self.skipTest("Run: python -m src.generate_dataset")
        result = parse_capture(path)
        self.assertIsInstance(result["bit_count"], int)

if __name__ == "__main__":
    unittest.main()
