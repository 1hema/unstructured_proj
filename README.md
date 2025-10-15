# Local RAG Chatbot — Streamlit + Ollama + Milvus Lite

A fully local  RAG chatbot that lets upload PDF file and query with their content.
 — powered by Ollama, Milvus Lite, and Streamlit.

No API keys, no cloud dependencies — everything runs on local machine.

---

## Features
-  Upload a PDF and automatically extract & chunk the text.
-  Store embeddings locally using Milvus Lite.
-  Use Ollama (`llama3`) as LLM for answering questions.
-  Simple and interactive Streamlit chat interface.
-  100% offline 

---

## Tech Stack
 Component  and  Tool 

LLM | [Ollama](https://ollama.ai) (`llama3`) 
Embeddings | `nomic-embed-text` (via Ollama) |
Vector Store | [Milvus Lite](https://milvus.io/docs/install_standalone-docker.md) |
UI | [Streamlit](https://streamlit.io) |
Framework | [LangChain](https://python.langchain.com/) |

Cmd to install dependencies--
pip3 install -r requirements.txt


