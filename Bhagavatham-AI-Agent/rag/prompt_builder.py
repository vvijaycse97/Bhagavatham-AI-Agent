"""
Prompt Builder.

Builds grounded prompts from a user query and filtered search results.

The PromptBuilder is intentionally independent of:
- Vector databases
- Embedding providers
- Retrieval implementations
- LLM providers

Author: Vijay V
Project: Bhagavatham AI Agent
"""

from __future__ import annotations

from models.search_result import SearchResult


DEFAULT_SYSTEM_INSTRUCTION = """
You are a Bhagavatham AI assistant.

Answer the user's question using only the provided context.

Rules:
- Use the provided context as the primary source of truth.
- Do not invent facts that are not supported by the context.
- If the answer cannot be determined from the context, clearly say that
  the available context does not contain enough information.
- Give a clear and concise answer.
- Preserve important names, places, events, and relationships accurately.
""".strip()


class PromptBuilder:
    """
    Builds a grounded prompt from retrieved search results.

    Responsibilities:
        - Validate and normalize the user query
        - Format retrieved context
        - Apply the system instruction
        - Build the final prompt

    The builder does not know about retrieval, vector databases,
    or LLM providers.
    """

    def __init__(
        self,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    ) -> None:
        """
        Initialize the PromptBuilder.

        Args:
            system_instruction:
                Instruction that defines how the LLM should answer.
        """

        if not system_instruction.strip():
            raise ValueError(
                "system_instruction must not be empty."
            )

        self._system_instruction = system_instruction.strip()

    def build(
        self,
        query: str,
        results: list[SearchResult],
    ) -> str:
        """
        Build a grounded LLM prompt.

        Args:
            query:
                User's question.

            results:
                Filtered and ranked search results.

        Returns:
            Fully formatted prompt.
        """

        normalized_query = self._normalize_query(query)

        context = self._build_context(results)

        return self._build_prompt(
            query=normalized_query,
            context=context,
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        """
        Normalize whitespace in the user query.

        Leading and trailing whitespace is removed.
        Consecutive whitespace characters are collapsed into
        a single space.

        Meaningful punctuation and Unicode characters are preserved.

        Args:
            query:
                Raw user query.

        Returns:
            Normalized query.

        Raises:
            ValueError:
                If the query is empty or contains only whitespace.
        """

        normalized = " ".join(query.split())

        if not normalized:
            raise ValueError(
                "query must not be empty."
            )

        return normalized

    def _build_context(
        self,
        results: list[SearchResult],
    ) -> str:
        """
        Format retrieved search results into context.
        """

        if not results:
            return (
                "No relevant context was retrieved."
            )

        context_sections: list[str] = []

        for index, result in enumerate(results, start=1):

            source = result.source.strip()

            if source:
                section = (
                    f"[Source {index}: {source}]\n"
                    f"{result.text.strip()}"
                )
            else:
                section = (
                    f"[Source {index}]\n"
                    f"{result.text.strip()}"
                )

            context_sections.append(section)

        return "\n\n".join(context_sections)

    def _build_prompt(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Construct the final grounded prompt.
        """

        return (
            f"{self._system_instruction}\n\n"
            "CONTEXT\n"
            "-------\n"
            f"{context}\n\n"
            "USER QUESTION\n"
            "-------------\n"
            f"{query}\n\n"
            "ANSWER\n"
            "------"
        )

    @property
    def system_instruction(self) -> str:
        """
        Return the configured system instruction.
        """

        return self._system_instruction