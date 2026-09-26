"""Question-shape validation: the deep-module (``_models.py``) parse seam.

Covers the five recurring mistakes from the 2026-09-24 ``trw_assess`` usage audit (worker-1, 38
calls): a ``"screen"`` type tag, ``criterion`` for
``criteria``, ``question`` for ``instructions``, an ``options`` list on a choice question, and
noul criteria keys that aren't exactly ``"true"``/``"false"``. Each must name the right field and
show a one-line fix, not a raw pydantic dump — and the fix must be enforced where
``DecisionQuestion`` is parsed, not as an ad-hoc check in the MCP tool.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trw_memory.decisions._models import (
    QUESTION_EXAMPLE,
    ChoiceQuestion,
    InvalidRequest,
    NoulQuestion,
    QuestionShapeError,
    ScoreQuestion,
    format_validation_error,
    parse_question,
    parse_questions,
)


class TestScreenIsNotAQuestionType:
    def test_screen_type_names_items_as_the_fix(self) -> None:
        with pytest.raises(QuestionShapeError, match=r"items=") as excinfo:
            parse_question("q", {"type": "screen", "instructions": "x"})
        assert "screen" in str(excinfo.value)

    def test_screen_type_is_an_invalid_request(self) -> None:
        """QuestionShapeError is a caller error like any other InvalidRequest."""
        assert issubclass(QuestionShapeError, InvalidRequest)

    def test_unknown_non_screen_type_names_the_valid_set(self) -> None:
        with pytest.raises(QuestionShapeError, match=r"'noul', 'choice' or 'score'"):
            parse_question("q", {"type": "verdict", "instructions": "x"})

    @pytest.mark.parametrize("unhashable", [["noul"], {"t": "noul"}])
    def test_unhashable_type_is_a_shape_error_not_a_type_error(self, unhashable: object) -> None:
        with pytest.raises(QuestionShapeError, match=r"'noul', 'choice' or 'score'"):
            parse_question("q", {"type": unhashable, "instructions": "x"})

    def test_missing_type_is_not_special_cased(self) -> None:
        """No 'type' key at all is a different failure (the union can't dispatch at all);
        it still surfaces, just via the ordinary pydantic path, not QuestionShapeError."""
        with pytest.raises(ValidationError):
            parse_question("q", {"instructions": "x"})

    @pytest.mark.parametrize("known_type", ["noul", "choice", "score"])
    def test_known_types_pass_the_tag_check(self, known_type: str) -> None:
        payloads = {
            "noul": {"type": "noul", "instructions": "x"},
            "choice": {"type": "choice", "instructions": "x", "criteria": {"a": "x"}},
            "score": {"type": "score", "instructions": "x", "criteria": ["lo", "hi"]},
        }
        parse_question("q", payloads[known_type])  # must not raise


class TestFieldRenames:
    def test_question_instead_of_instructions_names_the_fix(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            NoulQuestion.model_validate({"type": "noul", "question": "x"})
        assert "use 'instructions', not 'question'" in str(excinfo.value)

    def test_criterion_instead_of_criteria_names_the_fix(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ScoreQuestion.model_validate({"type": "score", "instructions": "x", "criterion": ["lo", "hi"]})
        assert "use 'criteria', not 'criterion'" in str(excinfo.value)

    def test_rename_is_skipped_when_the_right_field_is_also_present(self) -> None:
        """A caller who supplied BOTH keys gets pydantic's own extra-field error, not our rename
        message — the rename message would be misleading since 'instructions' is already there."""
        with pytest.raises(ValidationError) as excinfo:
            NoulQuestion.model_validate({"type": "noul", "question": "x", "instructions": "x"})
        assert "use 'instructions', not 'question'" not in str(excinfo.value)


class TestChoiceOptionsMistake:
    def test_options_list_names_criteria_as_the_fix(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ChoiceQuestion.model_validate({"type": "choice", "instructions": "x", "options": ["a", "b"]})
        message = str(excinfo.value)
        assert "'options' is not a field" in message and "criteria" in message

    def test_options_alongside_criteria_is_not_special_cased(self) -> None:
        """Once 'criteria' is present the extra 'options' key falls through to the ordinary
        extra-field error — still a caller error, just not our targeted message."""
        with pytest.raises(ValidationError) as excinfo:
            ChoiceQuestion.model_validate(
                {"type": "choice", "instructions": "x", "criteria": {"a": "x"}, "options": ["a"]}
            )
        assert "'options' is not a field" not in str(excinfo.value)


class TestNoulCriteriaKeys:
    @pytest.mark.parametrize("bad_keys", [{"yes": "y", "no": "n"}, {"true": "y", "maybe": "m"}, {"y": "y"}])
    def test_non_true_false_keys_are_rejected_with_the_offending_keys_named(self, bad_keys: dict[str, str]) -> None:
        with pytest.raises(ValidationError) as excinfo:
            NoulQuestion.model_validate({"type": "noul", "instructions": "x", "criteria": bad_keys})
        message = str(excinfo.value)
        assert "must be exactly 'true'/'false'" in message
        assert any(key in message for key in bad_keys)

    @pytest.mark.parametrize("ok_keys", [{"true": "y", "false": "n"}, {"true": "y"}, {"false": "n"}, None])
    def test_true_false_subset_or_absent_criteria_is_accepted(self, ok_keys: dict[str, str] | None) -> None:
        payload = {"type": "noul", "instructions": "x"}
        if ok_keys is not None:
            payload["criteria"] = ok_keys
        NoulQuestion.model_validate(payload)  # must not raise


class TestParseQuestionsAggregatesAndFormats:
    def test_parse_questions_returns_typed_questions_for_a_mixed_valid_batch(self) -> None:
        typed = parse_questions(
            {
                "a": {"type": "noul", "instructions": "x"},
                "b": {"type": "choice", "instructions": "x", "criteria": {"x": "x"}},
            }
        )
        assert typed["a"].type == "noul" and typed["b"].type == "choice"

    def test_format_validation_error_never_leaks_the_raw_pydantic_url_or_input_echo(self) -> None:
        questions = {"q": {"type": "choice", "instructions": "x", "criteria": {}}}
        try:
            parse_questions(questions)
        except ValidationError as exc:
            message = format_validation_error(exc, questions)
        else:
            raise AssertionError("expected a ValidationError")
        assert "questions.q" in message
        assert "Valid example: " + QUESTION_EXAMPLE in message
        # pydantic's own message includes a docs URL; the caller-facing one must not.
        assert "https://errors.pydantic.dev" not in message
