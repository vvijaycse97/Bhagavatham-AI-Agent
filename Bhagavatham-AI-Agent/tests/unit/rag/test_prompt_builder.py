"""
Unit tests for PromptBuilder.

Author: Vijay V
Project: Bhagavatham AI Agent
"""

from __future__ import annotations

import pytest

from models.search_result import SearchResult
from rag.prompt_builder import (
    DEFAULT_SYSTEM_INSTRUCTION,
    PromptBuilder,
)


class TestPromptBuilder:
    """Tests for PromptBuilder."""

    def test_build_prompt_with_results(self) -> None:
        """
        Prompt should contain system instruction,
        context, source, and user question.
        """

        builder = PromptBuilder()

        results = [
            SearchResult(
                chunk_id="chunk-1",
                text="Prahlada was a great devotee of Lord Vishnu.",
                source="Srimad Bhagavatham Part 7",
                similarity=0.92,
                rank=1,
            )
        ]

        prompt = builder.build(
            query="Who was Prahlada?",
            results=results,
        )

        assert DEFAULT_SYSTEM_INSTRUCTION in prompt
        assert "CONTEXT" in prompt
        assert "Prahlada was a great devotee of Lord Vishnu." in prompt
        assert "Srimad Bhagavatham Part 7" in prompt
        assert "USER QUESTION" in prompt
        assert "Who was Prahlada?" in prompt
        assert "ANSWER" in prompt

    def test_build_prompt_with_multiple_results(self) -> None:
        """Multiple results should be included in ranked order."""

        builder = PromptBuilder()

        results = [
            SearchResult(
                chunk_id="chunk-1",
                text="First context.",
                source="Part 1",
                rank=1,
            ),
            SearchResult(
                chunk_id="chunk-2",
                text="Second context.",
                source="Part 2",
                rank=2,
            ),
        ]

        prompt = builder.build(
            query="What happened?",
            results=results,
        )

        assert "First context." in prompt
        assert "Second context." in prompt
        assert "[Source 1: Part 1]" in prompt
        assert "[Source 2: Part 2]" in prompt

        assert prompt.index("First context.") < prompt.index(
            "Second context."
        )

    def test_build_prompt_without_results(self) -> None:
        """Prompt should handle an empty retrieval result."""

        builder = PromptBuilder()

        prompt = builder.build(
            query="Who is Krishna?",
            results=[],
        )

        assert "No relevant context was retrieved." in prompt
        assert "Who is Krishna?" in prompt
        assert "ANSWER" in prompt

    def test_empty_query_raises_error(self) -> None:
        """Empty query should raise ValueError."""

        builder = PromptBuilder()

        with pytest.raises(ValueError, match="query must not be empty"):
            builder.build(
                query="   ",
                results=[],
            )

    def test_empty_system_instruction_raises_error(self) -> None:
        """Empty system instruction should raise ValueError."""

        with pytest.raises(
            ValueError,
            match="system_instruction must not be empty",
        ):
            PromptBuilder(
                system_instruction="   "
            )

    def test_custom_system_instruction(self) -> None:
        """Custom system instruction should be preserved."""

        instruction = (
            "Answer only from the supplied Bhagavatham context."
        )

        builder = PromptBuilder(
            system_instruction=instruction
        )

        prompt = builder.build(
            query="Who is Arjuna?",
            results=[],
        )

        assert instruction in prompt
        assert builder.system_instruction == instruction

    def test_result_without_source(self) -> None:
        """Result without source should still be formatted."""

        builder = PromptBuilder()

        results = [
            SearchResult(
                chunk_id="chunk-1",
                text="Some Bhagavatham context.",
            )
        ]

        prompt = builder.build(
            query="What is this about?",
            results=results,
        )

        assert "[Source 1]" in prompt
        assert "Some Bhagavatham context." in prompt

    def test_query_leading_and_trailing_whitespace_is_removed(self) -> None:
        """Leading and trailing whitespace should be removed."""

        builder = PromptBuilder()

        prompt = builder.build(
            query="   Who is Krishna?   ",
            results=[],
        )

        assert "Who is Krishna?" in prompt
        assert "   Who is Krishna?   " not in prompt

    def test_query_multiple_spaces_are_normalized(self) -> None:
        """Consecutive spaces should be collapsed."""

        builder = PromptBuilder()

        prompt = builder.build(
            query="Who    is     Krishna?",
            results=[],
        )

        assert "Who is Krishna?" in prompt

    def test_query_tabs_and_newlines_are_normalized(self) -> None:
        """Tabs and newlines should become normal spaces."""

        builder = PromptBuilder()

        prompt = builder.build(
            query="Who\tis\nKrishna?",
            results=[],
        )

        assert "Who is Krishna?" in prompt

    def test_query_unicode_characters_are_preserved(self) -> None:
        """Unicode characters used in names should be preserved."""

        builder = PromptBuilder()

        prompt = builder.build(
            query="Who is Kṛṣṇa?",
            results=[],
        )

        assert "Who is Kṛṣṇa?" in prompt

    def test_query_punctuation_is_preserved(self) -> None:
        """Meaningful punctuation should not be removed."""

        builder = PromptBuilder()

        prompt = builder.build(
            query="Who was Prahlada's father?",
            results=[],
        )

        assert "Who was Prahlada's father?" in prompt

    def test_query_only_whitespace_raises_error(self) -> None:
        """Whitespace-only query should raise ValueError."""

        builder = PromptBuilder()

        with pytest.raises(ValueError, match="query must not be empty"):
            builder.build(
                query=" \t\n ",
                results=[],
            )