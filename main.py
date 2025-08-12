from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod


from models.state import AgentState, InputAgentState, OutputAgentState

from nodes.supervise import _supervise
from nodes.retrieve_documents import _retrieve_documents
from nodes.grade_documents import _grade_documents
from nodes.generate import _generate
from nodes.rephrase_query import _rephrase_query
from nodes.grade_answer import _grade_answer, GradingOutcome
from nodes.express_uncertainty import _express_uncertainty
from nodes.wrap_up import _wrap_up


def decide_to_generate(state: AgentState) -> bool:
    if (state.documents and len(state.documents) > 0) or len(
        state.rephrased_queries
    ) >= 3:
        return "generate"
    return "rephrase_query"


def get_graph() -> StateGraph:
    workflow = StateGraph(
        AgentState, input_schema=InputAgentState, output_schema=OutputAgentState
    )

    workflow.add_node("supervise", _supervise)
    workflow.add_node("retrieve_documents", _retrieve_documents)
    workflow.add_node("grade_documents", _grade_documents)
    workflow.add_node("generate", _generate)
    workflow.add_node("rephrase_query", _rephrase_query)
    workflow.add_node("express_uncertainty", _express_uncertainty)
    workflow.add_node("wrap_up", _wrap_up)

    workflow.add_edge(START, "supervise")

    workflow.add_conditional_edges(
        "supervise",
        tools_condition,
        {
            "tools": "retrieve_documents",
            END: END,
        },
    )

    workflow.add_edge("retrieve_documents", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        ["generate", "rephrase_query"],
    )

    workflow.add_edge("rephrase_query", "supervise")

    workflow.add_conditional_edges(
        "generate",
        _grade_answer,
        {
            GradingOutcome.NOT_SUPPORTED.value: "express_uncertainty",
            GradingOutcome.NOT_USEFUL.value: "rephrase_query",
            GradingOutcome.USEFUL.value: "wrap_up",
        },
    )

    workflow.add_edge("express_uncertainty", END)
    workflow.add_edge("wrap_up", END)

    return workflow.compile()


_graph = get_graph()


def main():

    # Draw the workflow diagram
    _graph.get_graph().draw_mermaid_png(
        output_file_path="resources/workflow_diagram.png",
        draw_method=MermaidDrawMethod.PYPPETEER,
        curve_style=CurveStyle.BASIS,
        max_retries=5,
        retry_delay=0.5,
    )


if __name__ == "__main__":
    main()
