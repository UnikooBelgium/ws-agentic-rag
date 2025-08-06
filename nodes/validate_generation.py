from models.state import AgentState
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from utils.aws_bedrock import chat_claude_4_sonnet


class GenerationValidation(BaseModel):
    is_answer_to_query: bool = Field(
        default=False,
        description="Indicates whether the generated answer is decent to the user's query.",
    )
    reasoning: str = Field(
        None, description="Explanation of the relevance check result."
    )


prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an expert in determining whether the generated answer is decent to the user's query.
The context is music production.
""",
        ),
        (
            "human",
            """
Your task is to analyze the generated answer and assess whether it is decent to the user's query.
Provide a brief explanation for your assessment.

- An answer is decent if it directly addresses the user's query and provides relevant information.
- An answer is decent when it says "I don't know" or "I don't have enough information to answer that question." if the query is not answerable.

User Query:,
{user_query}

Generated Answer:
{generated_answer}
""",
        ),
    ]
)


def _validate_generation(state: AgentState):
    """
    Check the relevance of the search results to the user's query.
    """
    user_query = state.original_user_query
    generation = state.generated_answer

    model_with_structured_output = chat_claude_4_sonnet.with_structured_output(
        GenerationValidation
    )
    response: GenerationValidation = (
        prompt_template | model_with_structured_output
    ).invoke({"user_query": user_query, "generated_answer": generation})

    return {
        "user_query_answered": response.is_answer_to_query,
        "user_query_answered_reasoning": response.reasoning,
    }
