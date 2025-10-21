# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from src.document_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

# Load environment (GROQ API key, model names, persist dir)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # MUST be set by developer
LLM_MODEL = os.getenv("LLM")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
PERSIST_DIR = os.getenv("PERSIST_DIR")

# Basic Streamlit UI
st.set_page_config(page_title="RAG Q&A", layout="centered")
st.title("📚 Document Q&A ")

st.write("Upload documents (PDF, TXT, CSV, DOCX, XLSX). The system will build a knowledge base and answer queries. Keys & models are hidden.")

# Upload area
uploaded = st.file_uploader("Upload documents", type=["pdf","txt","csv","docx","xlsx"], accept_multiple_files=True)
UPLOAD_DIR = "uploaded_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if uploaded:
    for f in uploaded:
        fp = os.path.join(UPLOAD_DIR, f.name)
        with open(fp, "wb") as out:
            out.write(f.read())
    st.success(f"Saved {len(uploaded)} file(s).")

# Build or Load index
if st.button("📦 Build / Load Knowledge Base"):
    vs = FaissVectorStore(persist_dir=PERSIST_DIR, embedding_model=EMBEDDING_MODEL)
    # try load existing
    if vs.load():
        st.success("Loaded existing knowledge base.")
    else:
        docs = load_all_documents(UPLOAD_DIR)
        if not docs:
            st.warning("No documents found in uploaded_data/ to index.")
        else:
            with st.spinner("Embedding and building index..."):
                vs.build_from_documents(docs)
            st.success("Knowledge base built and saved locally.")

# Querying
query = st.text_input("Ask a question about your documents:")
if st.button("Ask") and query.strip():
    vs = FaissVectorStore(persist_dir=PERSIST_DIR, embedding_model=EMBEDDING_MODEL)
    if not vs.load():
        st.error("No knowledge base found. Please upload documents and click Build first.")
    else:
        try:
            # retrieve top chunks
            results = vs.query(query, top_k=5)
            texts = [r["metadata"].get("text","") for r in results if r.get("metadata")]
            context = "\n\n".join(texts).strip()
            if not context:
                st.info("No relevant chunks found.")
            else:
                # prepare prompt to LLM
                prompt = f"Use the following context to answer the query: \"{query}\"\n\nContext:\n{context}\n\nAnswer concisely:"
                if not GROQ_API_KEY:
                    st.error("Server not configured with GROQ_API_KEY. Contact the admin.")
                else:
                    # call Groq LLM
                    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL)
                    with st.spinner("Generating answer..."):
                        resp = llm.invoke([prompt])  # response object from langchain_groq
                    # The exact attribute may vary; try common ones
                    answer = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
                    st.subheader("Answer")
                    st.write(answer)
                    
        except Exception as e:
            st.error(f"Error during query: {e}")
