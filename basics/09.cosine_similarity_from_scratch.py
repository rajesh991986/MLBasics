"""
=============================================================================
Q5: DOCUMENT SIMILARITY — INTERVIEW PRACTICE VERSION
Apple ML Engineer | Rights & Pricing Team | 60-min Technical Round
=============================================================================

PROBLEM: Find top-K most similar documents to a query using TF-IDF + cosine similarity.

=============================================================================
PHASE 1: CLARIFY (0-3 min) — ASK THESE BEFORE CODING
=============================================================================
SAY: "Before I start, let me clarify a few things:"

1. "Is this single query or batch queries?"
   → Code single first, mention batch matmul optimization later

2. "How many documents are we working with? Thousands? Millions?"
   → Affects algorithm choice: exact search vs approximate (FAISS)

3. "Is this running on-device (iPhone) or server-side?"
   → On-device: memory constraints, need sparse representation
   → Server: can use FAISS, pre-computed indexes

4. "Are documents pre-indexed or do we build TF-IDF each time?"
   → Production: pre-compute offline, only transform query at runtime

5. "What type of content? Single language or multilingual?"
   → Apple Rights & Pricing: 37+ languages, 175 countries

SAY: "Great, let me outline my approach."

=============================================================================
PHASE 2: APPROACH (3-5 min) — EXPLAIN BEFORE CODING
=============================================================================
SAY: "I'll build this in three parts:"

1. "TF-IDF vectorization — convert text to sparse numerical vectors"
   - Using sklearn TfidfVectorizer (mention: I can implement from scratch if needed)
   - Output is L2-normalized sparse matrix

2. "Cosine similarity — measure angle between query and each document vector"
   - Formula: cos(a,b) = (a · b) / (||a|| × ||b||)
   - Range: 0 (unrelated) to 1 (identical)
   - Why cosine over euclidean: "Captures direction/meaning, not magnitude.
     A short review and long review about same topic should be similar."

3. "Min-heap for top-K — efficiently track best K results"
   - O(n log k) instead of O(n log n) for full sort
   - Heap stores (score, index), smallest score on top
   - Only replace when we find something better

SAY: "Time complexity: O(n × d) for similarity + O(n log k) for top-K selection,
      where n = number of docs, d = vocab size, k = top_k."
SAY: "Space complexity: O(n × d) sparse for TF-IDF matrix + O(k) for heap."
SAY: "Does this approach sound good before I start coding?"

=============================================================================
PHASE 3: IMPLEMENT (5-25 min) — CODE WITH THINK-ALOUD
=============================================================================
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
import numpy as np


# --- SAY: "First, I'll write the cosine similarity function." ---
# --- SAY: "This computes the angle between two vectors in high-dimensional space." ---

def _cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two sparse TF-IDF vectors.

    Formula: cos(a, b) = (a · b) / (||a|| × ||b||)
    Range: 0.0 (orthogonal/unrelated) to 1.0 (identical direction)

    SAY: "I'm converting to dense here for clarity. In production,
          I'd use sparse operations or just matrix multiply — I'll show that after."
    """
    vec_a = vec_a.toarray().flatten()
    vec_b = vec_b.toarray().flatten()

    # SAY: "Computing L2 norms — need to guard against zero vectors"
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # SAY: "Zero norm means empty doc or all stopwords removed — return 0 similarity"
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return np.dot(vec_a, vec_b) / (norm_a * norm_b)

def _cosine_similarity_sparse(vec_a, vec_b):
    """Compute cosine similarity keeping sparse format."""
    # Dot product of sparse matrices
    dot_product = vec_a.multiply(vec_b).sum()
    
    # L2 norms using sparse operations
    norm_a = np.sqrt(vec_a.multiply(vec_a).sum())
    norm_b = np.sqrt(vec_b.multiply(vec_b).sum())
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def find_similar_docs_vectorized(query, docs, top_k=3, threshold=None):
    """Vectorized version - much faster for ML scale."""
    if not query or not docs:
        return []
    
    top_k = min(top_k, len(docs))
    if top_k <= 0:
        return []
    
    # Vectorize
    tfidf = TfidfVectorizer()
    doc_vectors = tfidf.fit_transform(docs)
    query_vector = tfidf.transform([query])
    
    # **SINGLE MATRIX MULTIPLY** - leverages BLAS, GPU-optimized
    # Since vectors are L2-normalized, dot product = cosine similarity
    similarities = (query_vector * doc_vectors.T).toarray().flatten()
    
    # Apply threshold (vectorized)
    if threshold is not None:
        mask = similarities >= threshold
        valid_indices = np.where(mask)[0]
        similarities = similarities[valid_indices]
        docs_array = np.array(docs)[valid_indices]
    else:
        valid_indices = np.arange(len(docs))
        docs_array = np.array(docs)
    
    # Get top-K using argpartition (O(n) on average)
    if top_k < len(similarities):
        # argpartition is O(n) average case vs O(n log n) for full sort
        top_k_indices = np.argpartition(similarities, -top_k)[-top_k:]
        # Only sort the top-k (O(k log k))
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
    else:
        top_k_indices = np.argsort(similarities)[::-1]
    
    results = [(docs_array[i], float(similarities[i])) for i in top_k_indices]
    return results

# --- SAY: "Now the main function. I'll use a min-heap for efficient top-K tracking." ---

def find_similar_docs(query, docs, top_k=3, threshold=None):
    """
    Find top-K most similar documents to query using TF-IDF + cosine similarity.

    Args:
        query:     Query string
        docs:      List of document strings
        top_k:     Number of top results to return
        threshold: Optional minimum similarity score (filter weak matches)

    Returns:
        List of (document, score) tuples, sorted by score descending
    """
    # --- SAY: "First, input validation — returning empty list for graceful pipeline behavior" ---
    # --- SAY: "In production, raising exceptions can break upstream services" ---
    if not query or not docs:
        return []

    top_k = min(top_k, len(docs))  # SAY: "Can't return more results than docs we have"
    if top_k <= 0:
        return []

    # --- SAY: "Vectorizing with TF-IDF. This builds vocabulary from docs, then transforms." ---
    # --- SAY: "sklearn's TfidfVectorizer L2-normalizes by default (norm='l2')" ---
    tfidf = TfidfVectorizer()
    doc_vectors = tfidf.fit_transform(docs)     # (n_docs, vocab_size) sparse matrix
    query_vector = tfidf.transform([query])      # (1, vocab_size) sparse matrix
    # SAY: "Using transform (not fit_transform) for query — same vocabulary as docs"

    # --- SAY: "Now I'll compute similarity and track top-K using a min-heap" ---
    # --- SAY: "Min-heap keeps the smallest score on top. If new score beats it, we swap." ---
    # --- SAY: "This is O(n log k) — better than sorting all O(n log n) when k << n" ---
    score_heap = []  # min-heap of (score, doc_index)

    for i in range(len(docs)):
        score = _cosine_similarity(query_vector, doc_vectors[i])

        # SAY: "Threshold filters out weak matches before they enter the heap"
        if threshold is not None and score < threshold:
            continue

        if len(score_heap) < top_k:
            heapq.heappush(score_heap, (score, i))
        elif score > score_heap[0][0]:
            # SAY: "heapreplace = pop smallest + push new in one operation — more efficient"
            heapq.heapreplace(score_heap, (score, i))

    # SAY: "Finally, sort the heap results by score descending for output"
    results = [(docs[i], float(score)) for score, i in sorted(score_heap, reverse=True)]
    return results


"""
=============================================================================
PHASE 4: TEST & EDGE CASES (25-35 min) — RUN THROUGH EXAMPLES
=============================================================================
SAY: "Let me test with a few cases to verify correctness."
"""

if __name__ == "__main__":

    # --- Test 1: Basic functionality ---
    # SAY: "Starting with a basic test — three docs, query about orders"
    docs = [
        "order is about prime membership",
        "order is in shipping state",
        "cat is sitting on the mat",
    ]
    query = "where is my order"

    print("=== Test 1: Basic ===")
    results = find_similar_docs(query, docs, top_k=2)
    for doc, score in results:
        print(f"  {score:.4f} | {doc}")
    # SAY: "As expected, both order-related docs score higher, cat doc filtered out"
    print()

    # --- Test 2: Larger dataset ---
    # SAY: "Testing with more docs to verify heap correctly keeps top-K"
    docs_large = [
        "track my package delivery status",
        "order is about prime membership",
        "how to return an item for refund",
        "order is currently in shipping state",
        "cat is sitting on the mat",
        "cancel my subscription immediately",
        "where is my recent purchase order",
        "reset my account password please",
    ]

    print("=== Test 2: Larger Dataset (top_k=3) ===")
    results = find_similar_docs("where is my order", docs_large, top_k=3)
    for doc, score in results:
        print(f"  {score:.4f} | {doc}")
    print()

    # --- Test 3: Threshold filtering ---
    # SAY: "Testing threshold — only return docs above 0.3 similarity"
    print("=== Test 3: Threshold = 0.3 ===")
    results = find_similar_docs("where is my order", docs, top_k=3, threshold=0.3)
    for doc, score in results:
        print(f"  {score:.4f} | {doc}")
    if not results:
        print("  No results above threshold")
    print()

    # --- EDGE CASES ---
    # SAY: "Now let me verify edge cases"

    # Edge Case 1: Empty input
    # SAY: "Empty query should return empty list, not crash"
    print("=== Edge Case: Empty Inputs ===")
    print(f"  Empty query:  {find_similar_docs('', docs)}")
    print(f"  Empty docs:   {find_similar_docs('test', [])}")
    print(f"  None query:   {find_similar_docs(None, docs)}")
    print()

    # Edge Case 2: top_k larger than docs
    # SAY: "top_k=100 but only 3 docs — should return all 3, not crash"
    print("=== Edge Case: top_k > len(docs) ===")
    results = find_similar_docs("order", docs, top_k=100)
    print(f"  Requested 100, got {len(results)} (correct: {len(docs)})")
    print()

    # Edge Case 3: Single document
    # SAY: "Single doc should still work"
    print("=== Edge Case: Single Doc ===")
    results = find_similar_docs("order shipping", ["order is shipping"], top_k=1)
    print(f"  {results}")
    print()

    # Edge Case 4: Query with only stopwords
    # SAY: "Stopword-only query gets empty TF-IDF vector — should return empty or zeros"
    print("=== Edge Case: Stopword-Only Query ===")
    results = find_similar_docs("the a an is", docs, top_k=2)
    print(f"  Results: {results}")
    # SAY: "All scores are 0.0 because query vector is empty after TF-IDF filtering"
    print()

    # Edge Case 5: Impossible threshold
    # SAY: "Threshold of 0.99 — nothing should match"
    print("=== Edge Case: Very High Threshold ===")
    results = find_similar_docs("order", docs, top_k=3, threshold=0.99)
    print(f"  Results: {results}")
    print()

    # Edge Case 6: Identical documents
    # SAY: "Duplicate docs should both appear with same score"
    print("=== Edge Case: Duplicate Docs ===")
    dup_docs = ["order tracking", "order tracking", "cat video"]
    results = find_similar_docs("order tracking", dup_docs, top_k=3)
    for doc, score in results:
        print(f"  {score:.4f} | {doc}")
    print()


"""
=============================================================================
PHASE 5: OPTIMIZATIONS — SAY THESE VERBALLY (35-45 min)
=============================================================================

SAY: "Now let me discuss production optimizations I'd make."

OPTIMIZATION 1: Sparse Matrix Multiply (replaces the loop)
─────────────────────────────────────────────────────────
SAY: "Since sklearn L2-normalizes TF-IDF vectors, cosine similarity
      equals dot product. I can replace the entire loop with one line:"

    scores = (doc_vectors @ query_vector.T).toarray().flatten()

SAY: "This uses sparse matrix multiplication — only touches non-zero elements.
      For 100K docs with 5K vocab, most entries are zero.
      Sparse matmul: O(nnz) vs dense loop: O(n × d). Massive speedup."

OPTIMIZATION 2: Batch Queries (matrix multiply generalizes)
─────────────────────────────────────────────────────────
SAY: "For multiple queries, the same matmul gives a score MATRIX:"

    query_vectors = tfidf.transform(queries)        # (n_queries, vocab)
    score_matrix = (query_vectors @ doc_vectors.T)   # (n_queries, n_docs)
    
SAY: "1000 queries processed in one operation — critical for batch processing
      at Apple's scale with 175 countries."

OPTIMIZATION 3: argpartition for Top-K (replaces heap)
─────────────────────────────────────────────────────────
SAY: "NumPy's argpartition finds top-K in O(n) average — even faster than heap's O(n log k):"

    top_k_idx = np.argpartition(scores, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

OPTIMIZATION 4: Pre-computed Index (production architecture)
─────────────────────────────────────────────────────────
SAY: "In production, I'd separate indexing from querying:"
    - Offline: fit TF-IDF on corpus, store doc_vectors + vocabulary
    - Online:  only transform query, compute similarity
    - "This avoids rebuilding vocabulary on every request"

OPTIMIZATION 5: Approximate Nearest Neighbors (10M+ scale)
─────────────────────────────────────────────────────────
SAY: "For 10M+ documents, exact search is too slow even with sparse matmul.
      I'd use FAISS with HNSW indexing:"
    - Build HNSW index offline: O(n log n)
    - Query time: O(log n) per query
    - Accuracy: ~98% recall at 10x speedup
    - "In my Amazon RAG system, we used OpenSearch with HNSW 
       to get P99 latency under 700ms for 400K+ articles"

=============================================================================
PHASE 6: APPLE CONTEXT — SAY THESE (45-50 min)
=============================================================================

SAY: "Let me connect this to Apple's Rights & Pricing context."

PRIVACY:
SAY: "For on-device similarity search — text never leaves the iPhone.
      Pre-compute TF-IDF vocabulary on server, ship to device.
      All similarity computation happens locally.
      Raw licensing documents stay on-device — only results are shown."

MULTILINGUAL (37+ languages):
SAY: "TF-IDF is language-agnostic for same-language matching,
      but cross-lingual needs multilingual embeddings.
      For Rights & Pricing: separate TF-IDF indexes per language group,
      with language detection routing queries to the right index.
      Similar to what I built at Amazon — language-specific BM25 indexes
      across 20 marketplaces."

ON-DEVICE DEPLOYMENT:
SAY: "For iPhone with 4GB RAM:"
    - Sparse matrix keeps memory low (100K docs × 5K vocab = ~40MB sparse vs ~4GB dense)
    - Quantize vocabulary weights from FP32 to INT8 (4x reduction)
    - Use Apple's Accelerate framework for optimized matrix operations
    - "Core ML can handle the vectorization on Neural Engine"

SCALE (Rights & Pricing specific):
SAY: "For matching licensing agreements across 175 countries:
      - Pre-compute document vectors for all contracts
      - Cluster similar contracts by region/content type
      - Use similarity search for duplicate clause detection
      - Flag conflicting terms across jurisdictions
      This is where my Amazon RAG experience directly applies —
      we indexed 400K+ articles with language-specific retrieval."

=============================================================================
COMPLEXITY SUMMARY — HAVE THIS READY
=============================================================================

Current Implementation:
    Time:  O(n × d) for TF-IDF + O(n × d) similarity loop + O(n log k) heap
    Space: O(n × d) sparse TF-IDF matrix + O(k) heap

Optimized (sparse matmul):
    Time:  O(nnz) matmul + O(n log k) top-K    (nnz << n × d for text)
    Space: O(nnz) sparse matrix + O(k) results

Production (FAISS/HNSW):
    Index:  O(n log n) build time (offline, one-time)
    Query:  O(log n) per query
    Space:  O(n × d') where d' = reduced dimensions

=============================================================================
PRACTICE CHECKLIST — Time yourself on each attempt
=============================================================================
Target times:
    [ ] Full implementation (no looking): < 20 minutes
    [ ] With tests and edge cases:        < 30 minutes  
    [ ] With verbal explanations:          < 40 minutes
    [ ] Complete with Apple context:       < 50 minutes

Practice until you can do this smoothly while talking through each step.
The code itself is ~40 lines. The verbal explanation is what makes it senior-level.
=============================================================================
"""