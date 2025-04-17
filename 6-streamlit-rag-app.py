import os
from pathlib import Path
from tempfile import mkdtemp

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import ExportType, DoclingLoader
from docling.chunking import HybridChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
def _get_env_from_colab_or_os(key):
    try:
        from google.colab import userdata

        try:
            return userdata.get(key)
        except userdata.SecretNotFoundError:
            pass
    except ImportError:
        pass
    return os.getenv(key)

load_dotenv()

# Configuration
HF_TOKEN = _get_env_from_colab_or_os("HF_TOKEN")
FILE_PATH = ["tests/AR_2020_WEB2.pdf"]
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL_ID = os.getenv("GEN_MODEL_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1")
EXPORT_TYPE = ExportType.DOC_CHUNKS
PROMPT = PromptTemplate.from_template(
    """
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {input}
Answer:
"""
)
TOP_K = int(os.getenv("TOP_K", 3))

# Cache expensive initialization
@st.cache_resource
def init_rag_pipeline():
    # 1. Load documents via DoclingLoader
    loader = DoclingLoader(
        file_path=FILE_PATH,
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
    )
    docs = loader.load()

    # 2. Determine splits
    if EXPORT_TYPE == ExportType.DOC_CHUNKS:
        splits = docs
    else:
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ('#', 'Header_1'),
                ('##', 'Header_2'),
                ('###', 'Header_3'),
            ],
        )
        splits = [chunk for doc in docs for chunk in splitter.split_text(doc.page_content)]

    # 3. Create embeddings and ingest into Milvus
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    milvus_uri = str(Path(mkdtemp()) / "streamlit_docling.db")
    vectorstore = Milvus.from_documents(
        documents=splits,
        embedding=embedding,
        collection_name="docling_streamlit",
        connection_args={"uri": milvus_uri},
        index_params={"index_type": "FLAT"},
        drop_old=True,
    )

    # 4. Setup RAG chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = HuggingFaceEndpoint(
        repo_id=GEN_MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
    )
    qa_chain = create_stuff_documents_chain(llm, PROMPT)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain

# Initialize app
def main():
    st.title("📚 PDF Q&A with RAG")

    # Initialize or retrieve chain
    rag_chain = init_rag_pipeline()

    # Session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Handle user input
    if user_input := st.chat_input("Ask anything about the PDF..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})

        # Run RAG chain
        with st.spinner("Retrieving and generating answer..."):
            resp = rag_chain.invoke({"input": user_input})

        answer = resp.get('answer', '').strip()

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
