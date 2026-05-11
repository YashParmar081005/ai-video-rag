import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

SMART_SYSTEM_PROMPT = """You are an expert assistant for video and meeting transcripts.

Answer the user's question using the transcript context below.
- If the answer is clearly stated, quote or summarize it directly.
- If the answer can be reasonably inferred from the context, provide that inference and note it is inferred.
- Only say "I could not find this information in the transcript" if there is truly NO relevant information at all.
- Be concise. Numbers, names, and facts should match the transcript exactly.

Transcript context:
{context}"""


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)

    # k=8 with MMR gives better coverage + diversity than k=4 similarity
    retriever = get_retriever(vector_store, k=8)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SMART_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # Full LCEL RAG pipeline
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt | llm | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SMART_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt | llm | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer
