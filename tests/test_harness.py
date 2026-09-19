from unittest.mock import patch

from harness.detection import DetectionResult
from harness.harness import Route, run


FAKE_SOURCES = [{"text": "Policy POL-00001 has coverage $50,000.", "metadata": {"source": "policies"}, "distance": 0.1}]


@patch("harness.harness.retrieve", return_value=FAKE_SOURCES)
@patch("harness.harness.claude_generate", return_value="The policy has coverage of $50,000 [1].")
@patch("harness.harness.confidence_signal", return_value=DetectionResult("confidence_signal", False, 0.0, {}))
@patch("harness.harness.groundedness_check", return_value=DetectionResult("groundedness", False, 0.0, {"claims": []}))
@patch("harness.harness.self_consistency_check", return_value=DetectionResult("self_consistency", False, 0.0, {}))
def test_clean_answer_passes(mock_sc, mock_ground, mock_conf, mock_gen, mock_retrieve):
    result = run("What is the coverage for POL-00001?", log=False)
    assert result.route == Route.PASS


@patch("harness.harness.retrieve", return_value=FAKE_SOURCES)
@patch("harness.harness.claude_generate", return_value="The policy has coverage of $999,999,999 [1].")
@patch("harness.harness.confidence_signal", return_value=DetectionResult("confidence_signal", False, 0.0, {}))
@patch("harness.harness.groundedness_check", return_value=DetectionResult(
    "groundedness", True, 0.6, {"claims": [{"claim": "coverage $999,999,999", "supported": False}]}
))
def test_badly_grounded_answer_falls_back(mock_ground, mock_conf, mock_gen, mock_retrieve):
    result = run("What is the coverage for POL-00001?", log=False)
    assert result.route == Route.FALLBACK
    assert "don't have enough verified information" in result.answer