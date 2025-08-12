from models.state import AgentState
from langchain_core.messages import AIMessage


def _wrap_up(state: AgentState) -> AgentState:
    """
    Wrap up the conversation and provide the final response.
    """
    generated_message = state.generated_answer
    return {
        "messages": [AIMessage(content=generated_message)],
        "answer": generated_message,
    }
