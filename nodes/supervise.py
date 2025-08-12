from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from models.state import AgentState
from utils.aws_bedrock import chat_claude_4_sonnet
from nodes.retrieve_documents import retriever_tool

prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a music production assistant supervisor that intelligently routes user queries to the appropriate response method.

Your primary responsibility is to analyze user queries and determine the best approach:

1. **For music production queries that would benefit from detailed technical information:**
   - Use the electronic_music_production_guide tool to search comprehensive resources covering:
     - Creative strategies and workflow techniques
     - Sound design and audio engineering concepts
     - Music theory applications in electronic music
     - Technical production solutions and troubleshooting
     - Composition, arrangement, and mixing techniques

2. **For simple music production questions or general inquiries:**
   - Provide direct, helpful responses based on your music production knowledge
   - Keep answers concise but informative

3. **For non-music production topics:**
   - Respond with "I don't know" to maintain focus

**Decision criteria for tool usage:**
- Use the tool when the query requires specific technical details, step-by-step processes, or comprehensive explanations
- Respond directly for basic definitions, quick tips, or when you can provide a complete answer immediately
- Consider the complexity and depth of information needed to properly address the user's question

Always prioritize providing the most helpful and accurate response for music producers.
""",
        ),
        (
            "human",
            """
USER QUESTION: {user_query}
FULL CHAT HISTORY: {chat_history}
""",
        ),
    ]
)


def _supervise(state: AgentState) -> AgentState:
    """
    Supervise the agent's actions and ensure they align with the user's intent.
    """

    query = state.original_user_query
    original_user_query = None

    if state.rephrased_queries and len(state.rephrased_queries) > 0:
        query = state.rephrased_queries[-1]
    elif len(state.messages) > 0 and isinstance(state.messages[-1], HumanMessage):
        query = state.messages[-1].content
        original_user_query = query

    tool_binded_model = chat_claude_4_sonnet.bind_tools([retriever_tool])

    response = (prompt_template | tool_binded_model).invoke(
        {
            "user_query": query,
            "chat_history": state.messages,
        }
    )

    return {"original_user_query": original_user_query, "messages": [response]}
