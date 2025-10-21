# 📄 Retrieval-Augmented Generation (RAG) App

## 🚀 Overview

This project is a **Retrieval-Augmented Generation (RAG) system** that enables users to upload documents, convert them into embeddings, store them using a vector database (FAISS), and query them using a powerful Large Language Model (LLM). The system is built using **LangChain**, **FAISS**, **SentenceTransformers**, and **Streamlit** for a user-friendly interface.

---

## 📁 Project Structure

```
my_rag_app/
│── app.py                # Streamlit interface
│── .env                  # Environment variables (API keys, model names)
│── src/
│   ├── document_loader.py # Loads PDF, TXT, CSV, DOCX, XLSX, JSON files
│   ├── embedding.py       # Splits documents and generates embeddings
│   ├── vectorstore.py     # Stores and retrieves embeddings using FAISS
│   └── rag_search.py      # Combines retrieval and LLM to answer questions
│── uploaded_data/         # Auto-created folder for uploaded files
│── faiss_store/           # Stores FAISS index and metadata
```

---

## 🧠 Component Descriptions

### 1. `document_loader.py`

**Purpose:** Load and convert documents into a unified structure.

* Supports PDF, TXT, CSV, DOCX, Excel, and JSON
* Returns a list of LangChain Document objects

### 2. `embedding.py`

**Purpose:** Transform document text into AI-understandable embeddings.

* Uses SentenceTransformer (default: `all-MiniLM-L6-v2`)
* Splits large documents into chunks
* Generates vector embeddings

### 3. `vectorstore.py`

**Purpose:** Store embeddings for fast search.

* Uses FAISS for similarity search
* Saves embeddings in `faiss.index`
* Saves text metadata in `metadata.pkl`

### 4. `rag_search.py`

**Purpose:** Perform RAG-style search and answer generation.

* Retrieves top relevant chunks
* Sends context to the LLM (Groq API)
* Returns summarized, human-like answers

### 5. `app.py`

**Purpose:** User-friendly interface built with Streamlit.

* Upload documents
* Build knowledge base
* Ask natural language questions
* View AI-generated answers

---

## 🔐 Environment Setup

Create a `.env` file with the following:

```
GROQ_API_KEY=your_key_here
LLM_MODEL=gemma2-9b-it
EMBEDDING_MODEL=all-MiniLM-L6-v2
PERSIST_DIR=faiss_store
```

---

## ▶️ How to Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 💡 How It Works

1. User uploads documents
2. Documents are chunked and embedded
3. Embeddings are stored in FAISS
4. User asks a question
5. Relevant document chunks are retrieved
6. LLM generates a summarized answer based on retrieved content

---

## 📦 Storage Details

| File           | Description                     |
| -------------- | ------------------------------- |
| `faiss.index`  | Stores numeric embeddings       |
| `metadata.pkl` | Stores text chunks and metadata |

---

## 🛠 Technologies Used

* **Streamlit** – Frontend interface
* **LangChain** – Document processing & RAG architecture
* **SentenceTransformers** – Embedding generation
* **FAISS** – Vector search
* **Groq LLM** – Answer generation

---

## 🌟 Features

✅ Multi-format document support
✅ Fast semantic search with FAISS
✅ Retrieval-Augmented responses
✅ Clean, minimal UI
✅ Secure API key handling via `.env`

---

## 📌 Future Enhancements

* Support chat history
* Add multi-user session storage
* Deploy on cloud platforms

---

## 🤝 Contributions

Feel free to fork this project and submit pull requests to improve features or documentation.

---

## 📞 Contact

For questions or improvements, contact the developer or open an issue.

---

**⭐ If you like this project, don’t forget to star the repo!**

# RAG Application using LangChain and Hugging Face

## 📌 Overview

This project is a Retrieval-Augmented Generation (RAG) application that allows users to upload documents, embed them using a pre-configured model, and retrieve accurate responses based on document content **without exposing API keys, model names, or vector store details to the end user**.

---

## 🚀 Features

* ✅ User-friendly Streamlit interface
* ✅ Secure embedding pipeline (no API key input required)
* ✅ Local vectorstore for fast retrieval
* ✅ Hugging Face or local embedding models
* ✅ Supports PDF, TXT, DOCX uploads
* ✅ Automatic chunking and indexing of documents

---

## 📂 Project Structure

| File                 | Description                                           |
| -------------------- | ----------------------------------------------------- |
| `app.py`             | Main Streamlit application for user interface         |
| `document_loader.py` | Loads and processes user-uploaded documents           |
| `embedding.py`       | Handles text embedding using a predefined model       |
| `vectorstore.py`     | Stores embedded document chunks using FAISS or Chroma |
| `utils.py`           | Helper functions                                      |
| `README.md`          | Documentation for the project                         |

---

## 🔧 How It Works

### 1. **User uploads a document**

The document is read and split into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.

### 2. **Embeddings are generated**

Chunks are embedded using a model defined in the backend (not visible to the user).

### 3. **Vectorstore is created**

Embeddings are stored locally using FAISS or Chroma for fast similarity search.

### 4. **User enters a query**

The query is embedded and matched against the stored vectors to retrieve relevant chunks.

### 5. **Answer generation**

The system combines retrieved information with an LLM to generate a factual, accurate response.

---

## ▶️ Running the Application

### **Prerequisites**

Make sure you have Python 3.10+ installed.

### **Install dependencies**

```bash
pip install -r requirements.txt
```

### **Start the app**

```bash
streamlit run app.py
```

---

## 🔐 No API Key Visible to User

* The API key is stored securely in the `.env` file or system environment variables.
* The user never sees or inputs the API key.

Example (`.env` file):

```env
GROQ_API_KEY=your_key_here
```

---

## 📊 Where Are Documents Stored?

* After embedding, files are stored in a local directory such as `vectorstore/`.
* Each document is saved as part of a FAISS or Chroma index.

Example structure:

```
/vectorstore
    index.faiss
    index.pkl
```

---

## ✨ Future Enhancements

* Add chat history memory
* Deploy to cloud (Streamlit Cloud / Hugging Face Spaces)
* Add authentication

---

## 📝 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open an issue to discuss changes.

---

## 🙌 Acknowledgements

Built with ❤️ using LangChain, Hugging Face, and FAISS.
