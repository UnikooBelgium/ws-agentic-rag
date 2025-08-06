from models.state import AgentState
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from utils.aws_bedrock import chat_claude_4_sonnet


class RelevanceCheck(BaseModel):
    search_results_relevant: bool = Field(
        default=False,
        description="Indicates whether the search results are relevant to the user's query.",
    )
    reasoning: str = Field(
        None, description="Explanation of the relevance check result."
    )


prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an expert in determining the relevance of search results to a user's query.
The context is music production.
""",
        ),
        (
            "human",
            """
Your task is to analyze the search results and assess their relevance to the user's query.
Provide a brief explanation for your assessment.

User Query:,
{user_query}

Search Results:
{search_results}
""",
        ),
    ]
)


def _check_relevance(state: AgentState):
    """
    Check the relevance of the search results to the user's query.
    """
    user_query = state.original_user_query
    search_results = state.search_results

    model_with_structured_output = chat_claude_4_sonnet.with_structured_output(
        RelevanceCheck
    )
    response: RelevanceCheck = (prompt_template | model_with_structured_output).invoke(
        {"user_query": user_query, "search_results": search_results}
    )

    return {
        "search_results_relevant": response.search_results_relevant,
        "relevance_reasoning": response.reasoning,
        "rephrased_queries": state.rephrased_queries,
    }
