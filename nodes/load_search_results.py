from models.state import AgentState
from utils.vector_store import _loaded_vector_store


def _load_search_results(state: AgentState):
    """
    Load search results into the agent state.
    """
    vector_store = _loaded_vector_store

    query = state.original_user_query
    if state.rephrased_queries and len(state.rephrased_queries) > 0:
        query = state.rephrased_queries[-1]

    search_results = vector_store.invoke(query)

    return {
        "search_results": [doc.page_content for doc in search_results],
        "search_results_relevant": False,
    }
