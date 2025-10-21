# document_loader.py
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load supported files from data_dir and return a list of LangChain Document objects.
    Supports: PDF, TXT, CSV, DOCX, XLSX
    """
    data_path = Path(data_dir).resolve()
    documents = []

    # Helper to glob by extension
    def _glob(ext):
        return list(data_path.glob(f"**/*.{ext}"))

    # PDF
    for f in _glob("pdf"):
        try:
            documents.extend(PyPDFLoader(str(f)).load())
        except Exception as e:
            print(f"[document_loader] PDF load error {f}: {e}")

    # TXT
    for f in _glob("txt"):
        try:
            documents.extend(TextLoader(str(f)).load())
        except Exception as e:
            print(f"[document_loader] TXT load error {f}: {e}")

    # CSV
    for f in _glob("csv"):
        try:
            documents.extend(CSVLoader(str(f)).load())
        except Exception as e:
            print(f"[document_loader] CSV load error {f}: {e}")

    # XLSX
    for f in _glob("xlsx"):
        try:
            documents.extend(UnstructuredExcelLoader(str(f)).load())
        except Exception as e:
            print(f"[document_loader] XLSX load error {f}: {e}")

    # DOCX
    for f in _glob("docx"):
        try:
            documents.extend(Docx2txtLoader(str(f)).load())
        except Exception as e:
            print(f"[document_loader] DOCX load error {f}: {e}")

    print(f"[document_loader] Loaded total documents: {len(documents)}")
    return documents
