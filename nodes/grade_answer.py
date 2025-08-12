from enum import Enum
from models.state import AgentState
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from utils.aws_bedrock import chat_claude_4_sonnet


class GradingOutcome(str, Enum):
    USEFUL = "useful"
    NOT_USEFUL = "not useful"
    NOT_SUPPORTED = "not supported"


class GradingResult(BaseModel):
    grading: bool = Field(
        default=False,
        description="Indicates whether the grading is positive or negative.",
    )


hallucination_grading_prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a grader assessing whether an LLM generation is grounded in and supported by a set of retrieved facts, specifically in the music production domain.

Your evaluation criteria:
- GROUNDED: The answer is directly supported by the provided facts/documents
- GROUNDED: The answer appropriately states "I don't know" when information is insufficient
- GROUNDED: The answer combines retrieved facts with well-established music production knowledge
- NOT GROUNDED: The answer contains claims not supported by the retrieved facts
- NOT GROUNDED: The answer provides specific details not present in the documents
- NOT GROUNDED: The answer contradicts the retrieved information

Assessment guidelines:
- Check if all factual claims in the answer can be traced back to the retrieved documents
- Verify that technical music production details are accurate and supported
- Ensure the answer directly addresses the user's query using available information
- Consider whether the response appropriately acknowledges limitations when information is incomplete
""",
        ),
        (
            "human",
            """
Assess whether this generated answer is grounded in and supported by the retrieved facts.

RETRIEVED FACTS/DOCUMENTS: {documents}

GENERATED ANSWER: {generated_answer}

Determine if the answer is grounded in the retrieved facts.
""",
        ),
    ]
)

answer_grading_prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a grader assessing whether an answer addresses and resolves a question, specifically in the music production domain.

Your evaluation criteria:
- ADDRESSES & RESOLVES: The answer directly responds to the question and provides actionable information
- ADDRESSES & RESOLVES: The answer appropriately states "I don't know" when the question cannot be answered
- ADDRESSES & RESOLVES: The answer provides sufficient detail to satisfy the user's information need
- ADDRESSES & RESOLVES: The answer stays focused on the specific question asked

- DOES NOT ADDRESS: The answer is off-topic or irrelevant to the question
- DOES NOT ADDRESS: The answer is too vague or generic to be useful
- DOES NOT ADDRESS: The answer avoids the question or provides unrelated information
- DOES NOT RESOLVE: The answer partially addresses the question but leaves key aspects unanswered

Assessment guidelines:
- Consider whether a music producer would find this answer helpful for their specific question
- Evaluate if the answer provides enough information to resolve the user's need
- Check if the answer maintains focus on the core question being asked
- Assess whether the response appropriately handles unanswerable questions
""",
        ),
        (
            "human",
            """
Assess whether this answer adequately addresses and resolves the user's question.

USER QUESTION: {user_query}

GENERATED ANSWER: {generated_answer}

Determine if the answer addresses and resolves the question, and provide reasoning for your assessment.
""",
        ),
    ]
)


def _grade_answer(state: AgentState):
    """
    Check whether the generated answer is grounded in the retrieved documents and addresses the user's query.
    """
    generation = state.generated_answer
    documents = state.documents
    user_query = state.original_user_query

    model_with_structured_output = chat_claude_4_sonnet.with_structured_output(
        GradingResult
    )

    chain = hallucination_grading_prompt_template | model_with_structured_output
    hallucination_grading_response: GradingResult = chain.invoke(
        {
            "generated_answer": generation,
            "documents": documents,
        }
    )

    if hallucination_grading_response.grading:
        chain = answer_grading_prompt_template | model_with_structured_output
        answer_grading_result: GradingResult = chain.invoke(
            {
                "generated_answer": generation,
                "user_query": user_query,
            }
        )
        if answer_grading_result.grading:
            return GradingOutcome.USEFUL
        else:
            return GradingOutcome.NOT_USEFUL
    else:
        return GradingOutcome.NOT_SUPPORTED
