from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import List, Optional, Annotated, Sequence


class InputAgentState(BaseModel):
    """
    Input schema for the agent state, which includes the user query.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages] = []


class OutputAgentState(BaseModel):
    """
    Output schema for the agent state, which includes the search results and answer.
    """

    answer: Optional[str] = None


class AgentState(InputAgentState, OutputAgentState):
    """
    State for the agent, which includes the conversation history.
    """

    original_user_query: Optional[str] = None
    search_results: Optional[List[str]] = None
    search_results_relevant: bool = False
    relevance_reasoning: Optional[str] = None
    generated_answer: Optional[str] = None
    user_query_answered: bool = False
    user_query_answered_reasoning: Optional[str] = None
    rephrased_queries: List[str] = []
