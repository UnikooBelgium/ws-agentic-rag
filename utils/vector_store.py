import faiss
from typing import List
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

from utils.aws_bedrock import embeddings, compressor

embedding_length = len(embeddings.embed_query("index dimensions"))
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
        print(f"Page {page.metadata['page']}: {page.page_content[:50]}...")

    documents = text_splitter.create_documents([page.page_content for page in pages])

    return documents


def load_vector_store(file_path: str) -> ContextualCompressionRetriever:

    vector_store = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(embedding_length),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vector_store.add_documents(load_pdf(file_path))

    print(f"Vector store loaded")

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
    )

    return compression_retriever


_loaded_vector_store = load_vector_store("resources/MakingMusic_DennisDeSantis.pdf")
