"""
Q18 — TF-IDF & Semantic Similarity [HIGH PROBABILITY]
Target time: 20 min | Requires: scikit-learn, numpy

APPROACH (say this in the first 60 seconds):
"Two approaches here — TF-IDF and sentence embeddings. TF-IDF is fast and
interpretable, good for keyword overlap. Embeddings capture semantic meaning
so 'great location' and 'perfect spot' score high even with no shared words.
I'll build TF-IDF first as a baseline, then explain the embedding upgrade.
At scale neither approach does O(n^2) pair-wise scan — you use Approximate
Nearest Neighbor search (FAISS/OpenSearch kNN)."

CORE MATH:
  TF-IDF:
    TF(t, d)  = count(t in d) / len(d)
    IDF(t)    = log(N / df(t) + 1)          # N=num docs, df=doc freq
    TF-IDF    = TF * IDF
    cosine(A, B) = (A · B) / (||A|| * ||B||)

  Embedding cosine: same formula but vectors are dense (384-d, 768-d, etc.)

WHEN TO USE WHICH:
  TF-IDF     → keyword search, fast, no GPU, explainable
  Embeddings → semantic search, paraphrase detection, multi-lingual
  BM25       → improved TF-IDF (penalises very long docs) — production baseline
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────
# SECTION 1: TF-IDF Cosine Similarity (stdlib-safe baseline)
# ─────────────────────────────────────────────────────────────

texts_a = [
    "Great location, close to the city centre",
    "Terrible service, staff was rude and unhelpful",
    "Clean rooms and excellent breakfast included",
]
texts_b = [
    "Perfect spot, walking distance to main attractions",
    "Poor customer service experience throughout stay",
    "Spacious rooms with amazing buffet every morning",
]


def tfidf_similarity_matrix(list_a: list, list_b: list) -> np.ndarray:
    """
    Fit a TF-IDF vectorizer on the union of both lists, then compute
    cosine similarity between every pair (a_i, b_j).

    Returns: (len(a) x len(b)) similarity matrix.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",   # drop 'the', 'and', etc.
        ngram_range=(1, 2),     # unigrams + bigrams
    )
    all_texts = list_a + list_b
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    vecs_a = tfidf_matrix[:len(list_a)]
    vecs_b = tfidf_matrix[len(list_a):]
    return cosine_similarity(vecs_a, vecs_b)


def top_pairs(sim_matrix: np.ndarray, list_a: list, list_b: list, top_k: int = 3):
    """Return the top-k most similar (a_i, b_j) pairs."""
    pairs = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            pairs.append((sim_matrix[i, j], i, j))
    pairs.sort(reverse=True)
    return pairs[:top_k]


if __name__ == "__main__":
    print("=" * 60)
    print("SECTION 1: TF-IDF Cosine Similarity")
    print("=" * 60)

    sim = tfidf_similarity_matrix(texts_a, texts_b)
    print("\nSimilarity matrix (rows=A, cols=B):")
    print(np.round(sim, 3))

    print("\nTop matching pairs:")
    for score, i, j in top_pairs(sim, texts_a, texts_b):
        print(f"  [{score:.3f}]  A[{i}]: '{texts_a[i]}'")
        print(f"          B[{j}]: '{texts_b[j]}'")

    # ── Best match per row ──
    print("\nBest B match for each A:")
    for i, row in enumerate(sim):
        j = int(np.argmax(row))
        print(f"  A[{i}] → B[{j}]  score={row[j]:.3f}")
        print(f"    A: {texts_a[i]}")
        print(f"    B: {texts_b[j]}")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Sentence Embeddings (semantic — better quality)
# ─────────────────────────────────────────────────────────────
# Requires:  pip install sentence-transformers
# Model 'all-MiniLM-L6-v2' is 80 MB, downloads once, runs on CPU.
#
# Why embeddings beat TF-IDF here:
#   "great location" vs "perfect spot" → TF-IDF: 0.00 (no shared tokens)
#                                      → Embeddings: ~0.82 (semantically close)

def embedding_similarity(list_a: list, list_b: list) -> np.ndarray:
    """
    Encode both lists with a sentence transformer and return
    the cosine similarity matrix.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [skip] sentence-transformers not installed — pip install sentence-transformers")
        return None

    model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings
    emb_a = model.encode(list_a, normalize_embeddings=True)  # (n, 384)
    emb_b = model.encode(list_b, normalize_embeddings=True)  # (m, 384)
    # With normalised embeddings: cosine sim = dot product
    return emb_a @ emb_b.T


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SECTION 2: Sentence Embedding Similarity")
    print("=" * 60)

    sim_emb = embedding_similarity(texts_a, texts_b)
    if sim_emb is not None:
        print("\nSimilarity matrix (rows=A, cols=B):")
        print(np.round(sim_emb, 3))

        print("\nBest B match for each A (semantic):")
        for i, row in enumerate(sim_emb):
            j = int(np.argmax(row))
            print(f"  A[{i}] → B[{j}]  score={row[j]:.3f}")
            print(f"    A: {texts_a[i]}")
            print(f"    B: {texts_b[j]}")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Scale — what you say when asked "10M documents?"
# ─────────────────────────────────────────────────────────────
"""
INTERVIEWER: "This works for 6 texts. What at 10 million documents?"

YOUR ANSWER:
  "O(n²) exact search is out. You use Approximate Nearest Neighbor (ANN):

  1. Encode all documents into embeddings ONCE and index them.
  2. Use FAISS (Facebook AI Similarity Search):
       - IVF (Inverted File) index: clusters vectors, only searches nearby clusters
       - HNSW (Hierarchical Navigable Small World): graph-based, very fast recall
  3. At query time: encode query → ANN search → return top-k in milliseconds.

  This is exactly the retrieval step in RAG systems.

  Example with FAISS:
      import faiss
      index = faiss.IndexFlatIP(dim)       # exact inner product (cosine if normalised)
      index = faiss.IndexIVFFlat(quantizer, dim, nlist)  # approximate, faster
      index.add(embeddings)                # O(n) build
      D, I = index.search(query_vec, k)    # O(log n) query

  Managed options: Pinecone, OpenSearch k-NN plugin, pgvector (Postgres),
  Weaviate, Qdrant — all expose the same encode→index→query pattern."
"""
SCALE_ANSWER = """
Approach      | Build time | Query time | Notes
──────────────────────────────────────────────────────────
Exact (brute) | O(n)       | O(n·d)     | fine up to ~100k
FAISS IVF     | O(n)       | O(√n · d)  | good to 100M
FAISS HNSW    | O(n log n) | O(log n)   | fast, high recall
Pinecone      | managed    | ~10ms p99  | no infra overhead
"""
print(SCALE_ANSWER)


# ─────────────────────────────────────────────────────────────
# SECTION 4: TF-IDF from scratch (if asked to implement)
# ─────────────────────────────────────────────────────────────

import math
from collections import Counter


def compute_tfidf_from_scratch(corpus: list) -> tuple:
    """
    Compute TF-IDF matrix without sklearn.
    Returns (tfidf_matrix as list-of-dicts, vocab).
    """
    # Tokenise
    tokenised = [doc.lower().split() for doc in corpus]
    N = len(tokenised)

    # Document frequency
    df = Counter()
    for tokens in tokenised:
        for term in set(tokens):
            df[term] += 1

    vocab = sorted(df.keys())

    tfidf_rows = []
    for tokens in tokenised:
        tf = Counter(tokens)
        row = {}
        for term in vocab:
            if tf[term] > 0:
                tf_val = tf[term] / len(tokens)
                idf_val = math.log((N + 1) / (df[term] + 1)) + 1  # smooth IDF
                row[term] = tf_val * idf_val
            else:
                row[term] = 0.0
        tfidf_rows.append(row)

    return tfidf_rows, vocab


def cosine_from_dicts(vec_a: dict, vec_b: dict) -> float:
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in vec_a)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    print("=" * 60)
    print("SECTION 4: TF-IDF from scratch")
    print("=" * 60)

    corpus = texts_a + texts_b
    rows, vocab = compute_tfidf_from_scratch(corpus)

    # cosine between first two docs
    score = cosine_from_dicts(rows[0], rows[3])
    print(f"\nFrom-scratch cosine (A[0] vs B[0]): {score:.4f}")
    print(f"  A: {corpus[0]}")
    print(f"  B: {corpus[3]}")

    print("\nAll sections complete.")
