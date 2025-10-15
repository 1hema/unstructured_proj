import streamlit as st
from rag_pipeline import build_rag_pipeline
import tempfile
import os

st.set_page_config(page_title="RAG PDF Chatbot (Milvus)", page_icon="🤖")
st.title("📄 Chat Bot")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Upload section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# check API key in .env 

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY == "":
    st.text_input("Enter your OpenAI API Key", type="password")

# Upload file 
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.write("🔍 Processing PDF with Milvus...")
    try:
        st.session_state.qa_chain = build_rag_pipeline(pdf_path)
        st.success("✅ PDF processed successfully! You can now ask questions.")
    except Exception as e:
        st.error(f"❌ Error setting up RAG pipeline: {e}")

# Chat input
user_input = st.chat_input("Ask something about your PDF...")

if user_input and st.session_state.qa_chain:
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            #response = st.session_state.qa_chain({"query": user_input})
            response = st.session_state.qa_chain.invoke({"query": user_input})
            st.markdown(response["result"])
