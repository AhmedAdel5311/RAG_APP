# vectorstore.py
import os
import faiss
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "vectorstore", embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index_path = os.path.join(self.persist_dir, "faiss.index")
        self.meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = None
        self.metadata: List[dict] = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

    def build_from_documents(self, documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200):
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        if embeddings.size == 0:
            print("[vectorstore] No embeddings to add.")
            return
        self._init_index(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata = [{"text": c.page_content} for c in chunks]
        self.save()
        print(f"[vectorstore] Built index with {self.index.ntotal} vectors")

    def _init_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

    def save(self):
        if self.index is None:
            return
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[vectorstore] Saved index -> {self.index_path}, metadata -> {self.meta_path}")

    def load(self) -> bool:
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path)):
            print("[vectorstore] No saved index found.")
            return False
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[vectorstore] Loaded index with {self.index.ntotal} vectors")
        return True

    def query(self, query_text: str, top_k: int = 5) -> List[dict]:
        q_emb = self.model.encode([query_text]).astype("float32")
        if self.index is None:
            raise RuntimeError("Index not initialized. Build or load the vectorstore first.")
        D, I = self.index.search(q_emb, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results
