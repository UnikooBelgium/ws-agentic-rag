from models.state import AgentState
from langchain_core.messages import AIMessage


def _express_uncertainty(state: AgentState) -> AgentState:
    """
    Express uncertainty in the generated answer.
    This node is used to indicate that the generated answer may not be fully accurate or complete.
    """
    uncertainty_message = """
I'm not entirely sure about the accuracy of the information provided.
The sources do not provide enough context to ensure complete accuracy.
"""
    generated_message = state.generated_answer

    result = uncertainty_message + "\n\n" + generated_message

    return {
        "messages": [AIMessage(content=result)],
    }
