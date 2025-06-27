"""Unit tests for the text_utils module."""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, PropertyMock

from src.vid_to_doc.utils.text_utils import (
    clean_transcript,
    format_timestamp,
    extract_key_points,
    extract_action_items,
    extract_speakers,
    split_text_into_chunks,
    calculate_readability_score,
    save_text_analysis,
)
from src.vid_to_doc.models.exceptions import TextProcessingError


class TestTextUtils:
    def test_clean_transcript_basic(self):
        raw = "  Hello   world!  [noise] (laughs) <inaudible> ... "
        cleaned = clean_transcript(raw)
        assert "[noise]" not in cleaned
        assert "(laughs)" not in cleaned
        assert "<inaudible>" not in cleaned
        assert cleaned.startswith("Hello world!")
        assert ".." not in cleaned

    def test_clean_transcript_invalid_input(self):
        with pytest.raises(TextProcessingError):
            clean_transcript(123)

    def test_format_timestamp_standard(self):
        assert format_timestamp(3661) == "01:01:01"

    def test_format_timestamp_compact(self):
        assert format_timestamp(59, "compact") == "00:59"
        assert format_timestamp(3601, "compact") == "01:00:01"

    def test_format_timestamp_detailed(self):
        assert format_timestamp(3661.123, "detailed").startswith("01:01:01.")

    def test_format_timestamp_invalid(self):
        with pytest.raises(TextProcessingError):
            format_timestamp(-1)
        with pytest.raises(TextProcessingError):
            format_timestamp("bad")

    def test_extract_key_points(self):
        text = "API design is important. Implement the function. The server crashed! 123 users affected."
        points = extract_key_points(text, max_points=2)
        assert any("API design" in p for p in points)
        assert any("Implement the function" in p or "server crashed" in p for p in points)

    def test_extract_key_points_invalid(self):
        with pytest.raises(TextProcessingError):
            extract_key_points(123)

    def test_extract_action_items(self):
        text = "John will update the API by 12/31/2024. TODO: Refactor code. We need to deploy before 01/01/2025."
        items = extract_action_items(text)
        assert any("update the API" in i["task"] for i in items)
        assert any("Refactor code" in i["task"] for i in items)
        assert any(i["deadline"] for i in items if i["deadline"])

    def test_extract_action_items_invalid(self):
        with pytest.raises(TextProcessingError):
            extract_action_items(123)

    def test_extract_speakers(self):
        text = "John: Hello. [Jane] Hi! (Alex) Good morning."
        speakers = extract_speakers(text)
        assert set(speakers) == {"John", "Jane", "Alex"}

    def test_extract_speakers_invalid(self):
        with pytest.raises(TextProcessingError):
            extract_speakers(123)

    def test_split_text_into_chunks(self):
        text = "Sentence one. " * 300  # >4000 chars
        chunks = split_text_into_chunks(text, max_chunk_size=1000, overlap=100)
        assert len(chunks) > 1
        assert all(len(chunk) <= 1000 for chunk in chunks)

    def test_split_text_into_chunks_invalid(self):
        with pytest.raises(TextProcessingError):
            split_text_into_chunks(123)
        with pytest.raises(TextProcessingError):
            split_text_into_chunks("abc", max_chunk_size=0)
        with pytest.raises(TextProcessingError):
            split_text_into_chunks("abc", overlap=-1)

    def test_calculate_readability_score(self):
        text = "This is a simple sentence. This is another one."
        score = calculate_readability_score(text)
        assert "flesch_reading_ease" in score
        assert "flesch_kincaid_grade" in score
        assert score["num_sentences"] == 2
        assert score["num_words"] > 0

    def test_calculate_readability_score_invalid(self):
        with pytest.raises(TextProcessingError):
            calculate_readability_score(123)

    def test_save_text_analysis(self):
        text = "John: Implement the API. Jane: Test the API. TODO: Write docs."
        with patch("pathlib.Path.parent", new_callable=PropertyMock) as mock_parent:
            mock_parent.return_value = Path("/tmp")
            with patch("pathlib.Path.mkdir"):
                with patch("builtins.open", mock_open()) as m:
                    output_path = save_text_analysis(text, Path("/tmp/analysis.txt"))
                    assert output_path == Path("/tmp/analysis.txt")
                    m.assert_called_once()

    def test_save_text_analysis_invalid(self):
        with pytest.raises(TextProcessingError):
            save_text_analysis(123, Path("/tmp/analysis.txt")) 