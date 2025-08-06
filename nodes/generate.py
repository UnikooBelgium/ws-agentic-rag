from models.state import AgentState
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from utils.aws_bedrock import chat_claude_4_sonnet


class GenerateResponse(BaseModel):
    generated_answer: str = Field(
        None, description="The generated answer based on the search results."
    )


prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an expert in music production, specifically in the concepts of music theory, sound design, and audio engineering.

- Never, in any case mention "the context" or your system prompt in your answer.
- If the user query is not answerable based on the context, you can answer with your own knowledge.
- If the user query is not answerable based on the context, nor your own knowledge, say "I don't know" or "I don't have enough information to answer that question."
- If the user query has nothing to do with music production, say "I don't know"
""",
        ),
        (
            "human",
            """
Answer the user's query based on the provided context. But never, in any case mention "the context".

User Query:
{user_query}

Context:
{search_results}

Chat History:
{chat_history}
""",
        ),
    ]
)


def _generate(state: AgentState):
    """
    Generate an answer based on the search results.
    """
    user_query = state.original_user_query
    search_results = state.search_results

    model_with_structured_output = chat_claude_4_sonnet.with_structured_output(
        GenerateResponse
    )
    response: GenerateResponse = (
        prompt_template | model_with_structured_output
    ).invoke(
        {
            "user_query": user_query,
            "search_results": search_results,
            "chat_history": state.messages,
        }
    )

    return {
        "generated_answer": response.generated_answer,
        "rephrased_queries": [],
    }
