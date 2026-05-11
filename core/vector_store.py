import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# How many chunks to return when querying
DEFAULT_K = 8
# MMR candidate pool — fetches more, then re-ranks for relevance + diversity
MMR_FETCH_K = 20


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def _clear_collection():
    """Clear old vector store data using Chroma API to avoid Windows file lock errors."""
    if os.path.exists(CHROMA_DIR):
        try:
            embeddings = get_embeddings()
            old_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR
            )
            old_store.delete_collection()
            print("Old vector store collection cleared via API.")
        except Exception as e:
            print(f"Could not clear collection: {e}")


def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store...")

    # Always rebuild fresh — avoids stale chunks from previous videos
    _clear_collection()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,   # slightly larger to keep more context per chunk
        chunk_overlap=100, # larger overlap to avoid splitting key sentences
    )
    chunks = splitter.split_text(transcript)
    print(f"  → {len(chunks)} chunks created from transcript.")

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print("Vector store built successfully.")
    return vector_store


def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def get_retriever(vector_store: Chroma, k: int = DEFAULT_K):
    """Use MMR retrieval for better relevance + diversity than pure similarity."""
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": MMR_FETCH_K},
    )
