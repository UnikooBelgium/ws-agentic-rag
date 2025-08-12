from models.state import AgentState
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from utils.aws_bedrock import chat_claude_4_sonnet


class RephraseResponse(BaseModel):
    rephrased_user_query: str = Field(
        None, description="The rephrased user query for better search results."
    )


prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a question re-writer that converts an input question to a better version optimized for vectorstore retrieval in the music production domain.

Your expertise includes:
- Music theory, sound design, and audio engineering concepts
- Understanding semantic intent and underlying meaning of queries
- Optimizing queries for better document matching and retrieval

Guidelines for rephrasing:
- Analyze the underlying semantic intent and meaning of the original question
- Use specific music production terminology when appropriate
- Include relevant synonyms and related concepts that might appear in documents
- Make queries more specific and searchable while preserving the original intent
- Consider different ways the same concept might be expressed in music production contexts
- Avoid overly broad or vague reformulations
- The result must always be formulated as a clear question.
""",
        ),
        (
            "human",
            """
Analyze the semantic intent of this music production question and rephrase it for optimal vectorstore retrieval.

ORIGINAL QUESTION: {user_query}

PREVIOUS REPHRASE ATTEMPTS: {rephrased_queries}

CURRENT RETRIEVED DOCUMENTS: {documents}

PREVIOUS RESPONSE GENERATED: {generated_answer}

Create a new rephrased version that captures the underlying meaning while using different terminology or structure to improve document retrieval.
""",
        ),
    ]
)


def _rephrase_query(state: AgentState):
    """
    Generate an answer based on the search results.
    """
    user_query = state.original_user_query
    documents = state.documents
    rephrased_queries = state.rephrased_queries
    generated_answer = state.generated_answer

    model_with_structured_output = chat_claude_4_sonnet.with_structured_output(
        RephraseResponse
    )
    response: RephraseResponse = (
        prompt_template | model_with_structured_output
    ).invoke(
        {
            "user_query": user_query,
            "rephrased_queries": rephrased_queries,
            "documents": documents,
            "generated_answer": generated_answer,
        }
    )

    rephrased_queries = state.rephrased_queries + [response.rephrased_user_query]

    return {
        "rephrased_queries": rephrased_queries,
    }
