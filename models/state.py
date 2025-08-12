from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import List, Optional, Annotated, Sequence


class InputAgentState(BaseModel):
    """
    Input schema for the agent state, which includes the user query.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages] = []


class OutputAgentState(InputAgentState):
    """
    Output schema for the agent state, which includes the search results and answer.
    """

    pass


class AgentState(InputAgentState):
    """
    State for the agent, which includes the conversation history.
    """

    original_user_query: Optional[str] = None
    documents: Optional[str] = None
    generated_answer: Optional[str] = None
    rephrased_queries: List[str] = []
