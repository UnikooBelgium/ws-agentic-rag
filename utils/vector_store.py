import os
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

from utils.aws_bedrock import embeddings, compressor

# Directory for persistent storage
PERSIST_DIRECTORY = ".chroma_db"

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=2048,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)


def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    pages: List[Document] = []
    for page in loader.lazy_load():
        pages.append(page)

        # Pretty print the page content
        print(f"Loading page {page.metadata['page']}: {page.page_content[:50]}...\n\n")

    documents = text_splitter.create_documents([page.page_content for page in pages])

    return documents


def load_vector_store(file_path: str) -> ContextualCompressionRetriever:
    # Create persist directory if it doesn't exist
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    # Initialize ChromaDB with persistence
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name="pdf_documents",
    )

    # Check if the collection already has documents
    if vector_store._collection.count() == 0:
        print("Loading PDF and creating embeddings...")
        documents = load_pdf(file_path)
        vector_store.add_documents(documents)
        print("Documents added to vector store")
    else:
        print("Vector store already contains documents")

    print(f"Vector store loaded with {vector_store._collection.count()} documents")

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_store.as_retriever(search_kwargs={"k": 20}),
    )

    return compression_retriever
