from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.graph import MermaidDrawMethod

from models.state import AgentState, InputAgentState, OutputAgentState

from nodes.user_intent import _user_intent
from nodes.load_search_results import _load_search_results
from nodes.check_relevance import _check_relevance
from nodes.generate import _generate
from nodes.rephrase_query import _rephrase_query
from nodes.validate_generation import _validate_generation
from nodes.wrap_up import _wrap_up


def should_continue_after_relevance(state: AgentState) -> bool:
    if state.search_results_relevant or len(state.rephrased_queries) >= 3:
        return "generate"
    return "rephrase_query"


def should_continue_after_generation(state: AgentState) -> bool:
    if state.user_query_answered:
        return "wrap_up"
    return "rephrase_query"


def get_graph() -> StateGraph:
    workflow = StateGraph(
        AgentState, input_schema=InputAgentState, output_schema=OutputAgentState
    )

    workflow.add_node("user_intent", _user_intent)
    workflow.add_node("load_search_results", _load_search_results)
    workflow.add_node("check_relevance", _check_relevance)
    workflow.add_node("generate", _generate)
    workflow.add_node("rephrase_query", _rephrase_query)
    workflow.add_node("validate_generation", _validate_generation)
    workflow.add_node("wrap_up", _wrap_up)

    workflow.add_edge(START, "user_intent")
    workflow.add_edge("user_intent", "load_search_results")
    workflow.add_edge("load_search_results", "check_relevance")
    workflow.add_edge("rephrase_query", "load_search_results")

    workflow.add_conditional_edges(
        "check_relevance",
        should_continue_after_relevance,
        ["generate", "rephrase_query"],
    )

    workflow.add_edge("generate", "validate_generation")

    workflow.add_conditional_edges(
        "validate_generation",
        should_continue_after_generation,
        ["wrap_up", "rephrase_query"],
    )

    workflow.add_edge("wrap_up", END)

    return workflow.compile()


_graph = get_graph()


def main():
    _graph.get_graph().draw_mermaid_png(
        output_file_path="resources/workflow_diagram.png",
    )


if __name__ == "__main__":
    main()
