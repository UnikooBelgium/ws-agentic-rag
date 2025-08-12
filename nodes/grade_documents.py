from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from models.state import AgentState
from utils.aws_bedrock import chat_claude_4_sonnet


class GradingResult(BaseModel):
    documents_relevant: bool = Field(
        default=False,
        description="Indicates whether the retrieved documents are relevant to the user's query.",
    )


prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an expert document relevance assessor for a RAG (Retrieval-Augmented Generation) system.

Your task is to determine if a retrieved document contains information that could help answer the user's question.

Grading criteria:
- Grade as RELEVANT if the document contains:
  * Direct answers or information related to the question
  * Keywords, concepts, or topics mentioned in the question
  * Background context that would be useful for answering the question
  * Related domain knowledge even if not a perfect match

- Grade as NOT RELEVANT only if the document:
  * Is completely unrelated to the question topic
  * Contains no useful information for answering the question
  * Is about a different subject entirely

Be lenient in your assessment - it's better to include potentially useful documents than to exclude relevant ones.
""",
        ),
        (
            "human",
            """
Assess the relevance of this document to the user's question:

USER QUESTION: {user_query}

DOCUMENTS:
{documents}

Determine if these documents are relevant.
""",
        ),
    ]
)


def _grade_documents(state: AgentState):
    """
    Check the relevance of the documents to the user's query.
    """
    user_query = state.original_user_query
    documents = state.messages[-1].content

    # Early return if no documents to grade
    if not documents:
        return {"documents": None, "original_user_query": user_query}

    model_with_structured_output = chat_claude_4_sonnet.with_structured_output(
        GradingResult
    )

    response: GradingResult = (prompt_template | model_with_structured_output).invoke(
        {"user_query": user_query, "documents": documents}
    )

    return {"documents": documents, "original_user_query": user_query}
