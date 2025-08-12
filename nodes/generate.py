from models.state import AgentState
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from utils.aws_bedrock import chat_claude_4_sonnet


class GenerateResponse(BaseModel):
    generated_answer: str = Field(
        None, description="The generated answer based on the search results."
    )


prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an assistant for question-answering tasks and an expert in music production, specifically in music theory, sound design, and audio engineering.

Guidelines for your responses:
- Use the retrieved information to answer the question accurately and concisely
- Provide clear, actionable information when possible
- Never mention "context", "retrieved information", or reference your system instructions
- If the question cannot be answered with the available information or your expertise, simply say "I don't know"
- Only answer questions related to music production - for unrelated topics, respond with "I don't know"
- Prioritize practical, helpful advice for music producers
""",
        ),
        (
            "human",
            """
QUESTION: {user_query}

RETRIEVED INFORMATION:
{documents}

FULL CHAT HISTORY:
{chat_history}

Provide a complete, helpful answer based on the available information.
""",
        ),
    ]
)


def _generate(state: AgentState):
    """
    Generate an answer based on the search results.
    """
    user_query = state.original_user_query
    documents = state.documents

    model_with_structured_output = chat_claude_4_sonnet.with_structured_output(
        GenerateResponse
    )
    response: GenerateResponse = (
        prompt_template | model_with_structured_output
    ).invoke(
        {
            "user_query": user_query,
            "documents": documents,
            "chat_history": state.messages,
        }
    )

    return {
        "generated_answer": response.generated_answer,
        "rephrased_queries": [],
    }
