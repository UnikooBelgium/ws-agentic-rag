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
You are an expert in rephrasing user queries for better search results or generating answers.
The context is music production.
""",
        ),
        (
            "human",
            """
Your task is to rephrase the user's query to improve search results or generate a more accurate answer.

User Query:
{user_query}

Already rephrased Queries:
{rephrased_queries}

Search Results:
{search_results}

Generated response:
{generated_answer}
""",
        ),
    ]
)


def _rephrase_query(state: AgentState):
    """
    Generate an answer based on the search results.
    """
    user_query = state.original_user_query
    search_results = state.search_results
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
            "search_results": search_results,
            "generated_answer": generated_answer,
        }
    )

    rephrased_queries = state.rephrased_queries + [response.rephrased_user_query]

    return {
        "rephrased_queries": rephrased_queries,
    }
