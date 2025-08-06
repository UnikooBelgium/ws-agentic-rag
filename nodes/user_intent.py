from models.state import AgentState
from langchain_core.messages import HumanMessage


def _user_intent(state: AgentState) -> AgentState:
    """
    Determine the user's intent based on the input query.
    """

    original_user_query = None

    if len(state.messages) > 0 and isinstance(state.messages[-1], HumanMessage):
        original_user_query = state.messages[-1].content

    return {"original_user_query": original_user_query}
