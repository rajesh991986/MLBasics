"""
Q9 — Document Chunking + Embedding Pipeline [ALMOST CERTAIN]
Target time: 25 min | CoderPad-safe (Python stdlib + NumPy)

APPROACH (say this first 60 seconds):
"I'll build a three-stage RAG indexing pipeline. First, chunk_document splits text into
overlapping chunks that respect sentence boundaries. Second, a mock embedding function
using bag-of-words normalized vectors (in production: sentence-transformers). Third, a
VectorStore class that stores chunk-embedding pairs and supports top-K retrieval via
cosine similarity. Time: O(n) for chunking, O(d) for embedding per chunk, O(n*d) for
brute-force search."

CORE MATH:
- Chunking: sentence-boundary aware with overlap for context preservation
- Embedding: BoW hashed to fixed dim, L2 normalized
- Retrieval: cosine similarity = dot(a,b) / (||a|| * ||b||)

TIME: O(n) chunk, O(chunks*d) search | SPACE: O(chunks * d)
"""

import re
import numpy as np


def chunk_document(text: str, chunk_size: int = 200, overlap: int = 50,
                   respect_sentences: bool = True) -> list:
    """Split text into overlapping chunks, respecting sentence boundaries."""
    if not text.strip():
        return []

    if not respect_sentences:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start += chunk_size - overlap
        return [c for c in chunks if c]

    # sentence-boundary-aware chunking
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sent_len = len(sentence)
        if current_length + sent_len > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # keep last sentence(s) for overlap
            overlap_sents = []
            overlap_text = ''
            for s in reversed(current_chunk):
                if len(overlap_text) + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_text = ' '.join(overlap_sents)
            current_chunk = overlap_sents
            current_length = len(overlap_text)

        current_chunk.append(sentence)
        current_length += sent_len + 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def mock_embed(text, dim=64):
    vec = np.zeros(dim)
    for token in text.lower().split():
        vec[hash(token) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class VectorStore:
    """Simple in-memory vector store for chunk retrieval."""

    def __init__(self, embed_fn=mock_embed):
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        self.embed_fn = embed_fn

    def add_document(self, text: str, chunk_size: int = 200,
                     overlap: int = 50, doc_id: str = "doc_0") -> int:
        chunks = chunk_document(text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.embeddings.append(self.embed_fn(chunk))
            self.metadata.append({'doc_id': doc_id, 'chunk_index': i})
        return len(chunks)

    def query(self, query_text: str, top_k: int = 3) -> list:
        if not self.chunks:
            return []
        query_emb = self.embed_fn(query_text)
        scored = []
        for i, emb in enumerate(self.embeddings):
            score = cosine_sim(query_emb, emb)
            scored.append({'chunk': self.chunks[i], 'score': round(score, 4),
                           'metadata': self.metadata[i]})
        scored.sort(key=lambda x: -x['score'])
        return scored[:top_k]

    def __len__(self):
        return len(self.chunks)


# === TEST ===
if __name__ == "__main__":
    contract_text = """
    This Distribution Agreement is entered into between Apple Inc.
    and Partner Corp effective January 1, 2025. The territory covered under this
    agreement includes the United States, Canada, United Kingdom, Germany, France,
    and Japan. The content licensed includes all music catalog items, podcasts,
    and audiobook titles. The royalty rate shall be 70% of net revenue for music
    and 50% for podcast content. Payment terms are net-30 from the end of each
    calendar quarter. This agreement shall remain in effect for a period of 3 years
    from the effective date, with automatic renewal unless terminated with 90 days
    written notice. Apple reserves the right to adjust pricing in any territory
    based on local market conditions, currency fluctuations, and regulatory requirements.
    """

    store = VectorStore()
    num_chunks = store.add_document(contract_text, chunk_size=200, overlap=50,
                                     doc_id="agreement_001")
    print(f"Indexed {num_chunks} chunks from document")
    print(f"Store size: {len(store)} chunks\n")

    queries = [
        "What territories are covered?",
        "What are the royalty rates?",
        "How long is the agreement valid?",
    ]
    for q in queries:
        print(f"Query: '{q}'")
        results = store.query(q, top_k=2)
        for r in results:
            print(f"  Score={r['score']:.4f}: {r['chunk'][:80]}...")
        print()

    # Edge case
    assert chunk_document("") == []
    print("All tests passed!")
