from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode

from models.state import AgentState
from utils.vector_store import load_vector_store

loaded_vector_store = load_vector_store("resources/MakingMusic_DennisDeSantis.pdf")
retriever_tool = create_retriever_tool(
    loaded_vector_store,
    "electronic_music_production_guide",
    """
    Search and return comprehensive information about electronic music production techniques, creative strategies, and solutions for common challenges faced by electronic music producers.
    This resource contains 74 detailed strategies covering how to start new tracks when facing creative blocks, generate musical ideas through various methods including catalog of attributes and active listening,
    choose appropriate sounds and tempos, organize workflow and studio setup, overcome procrastination and creative paralysis, program realistic drum beats with proper timing and groove,
    create compelling melodies using contour and motivic development, build effective harmony progressions from basic triads to extended jazz chords, develop bass lines and manage low-end frequency conflicts,
    apply sampling techniques both modern and vintage, use automation creatively for rhythmic interest, arrange songs using subtractive processes and formal structures,
    create tension and release through dramatic arc and textural density changes, finish tracks effectively with proper endings and know when to stop refining,
    and solve technical production issues including voice leading, tuning everything in the mix, managing silence and noise as compositional elements,
    and using randomization tools responsibly while maintaining creative control throughout the music-making process.
    """,
)

_retrieve_documents = ToolNode([retriever_tool])
