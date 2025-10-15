from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain_milvus import Milvus
import os
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def build_rag_pipeline(pdf_path, collection_name="pdf_chatbot"):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Use Ollama embeddings + LLM
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3")

    # Use Milvus Lite (file-based vector store)
  

    def get_vectorstore(embeddings):
        connection_args = {
            "uri": "./milvus_local.db",  # local lite db
        }

        # ✅ This is all you need — Milvus Lite automatically runs in sync mode
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args=connection_args,
            collection_name="pdf_docs",
            auto_id=True,
        )
        vectorstore.add_documents(chunks)

        print("✅ Milvus Lite initialized successfully (sync mode)")
        return vectorstore

    # usage
    vectorstore = get_vectorstore(embeddings)

    # Semantic retriever
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Keyword retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(chunks)

    # Combine both
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        #retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        retriever=hybrid_retriever,
        return_source_documents=True,
    )

    return qa_chain
