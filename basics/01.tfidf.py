import numpy as np
import re
from collections import Counter
from scipy.sparse import csr_matrix

class TFIDFVectorizer:
    def __init__(self, min_df=1, max_df=1.0, max_features=None):
        """
        TF-IDF Vectorizer from scratch.

        Args:
            min_df (int/float): Ignore terms with doc freq lower than this.
            max_df (int/float): Ignore terms with doc freq higher than this.
            max_features (int): Keep only the top N most frequent terms.
        """
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features

        # Model State
        self.vocab = {}  # word -> index
        self.idf_diag = None  # Pre-computed IDF vector
        self.vocab_size = 0

    def _tokenize(self, text):
        """Tokenize text using regex (2+ character words only)."""
        return re.findall(r'\b\w\w+\b', text.lower())

    def fit(self, documents):
        """
        Learn vocabulary and IDF from corpus.

        Time: O(N*M) where N=docs, M=avg_words
        Space: O(V) where V=unique_terms
        """
        # Input validation
        if documents is None:
            raise ValueError("documents cannot be None")
        if len(documents) == 0:
            self.vocab = {}
            self.idf_diag = np.array([])
            self.vocab_size = 0
            return self

        # 1. Global Term Counting
        doc_freqs = Counter()  # DF (Document Frequency)
        n_docs = len(documents)

        for doc in documents:
            if doc is None:
                continue
            unique_tokens = set(self._tokenize(doc))
            doc_freqs.update(unique_tokens)

        # 2. Apply Filtering (min_df / max_df)
        _min = self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs)
        _max = self.max_df if isinstance(self.max_df, int) else int(self.max_df * n_docs)

        filtered_words = []
        for word, count in doc_freqs.items():
            if _min <= count <= _max:
                filtered_words.append(word)

        # 3. Apply max_features (Sort by freq desc, take top N)
        if self.max_features is not None:
            sorted_words = sorted(filtered_words, key=lambda w: (-doc_freqs[w], w))
            filtered_words = sorted_words[:self.max_features]
        else:
            filtered_words = sorted(filtered_words)

        # 4. Freeze Vocabulary
        self.vocab = {word: i for i, word in enumerate(filtered_words)}
        self.vocab_size = len(self.vocab)

        # 5. Compute IDF Vector
        # idf = log(N / (df + 1)) + 1 (sklearn smooth_idf style)
        idf_values = []
        for word in filtered_words:
            df = doc_freqs[word]
            idf = np.log((n_docs + 1) / (df + 1)) + 1
            idf_values.append(idf)

        self.idf_diag = np.array(idf_values)

        return self

    def transform(self, documents):
        """
        Convert docs to Sparse CSR Matrix.

        Time: O(N*M) for tokenization + vocab lookup
        Space: O(nnz) where nnz=non-zero entries
        """
        if documents is None:
            raise ValueError("documents cannot be None")

        if self.vocab_size == 0:
            return csr_matrix((len(documents), 0))

        rows = []
        cols = []
        data = []

        for row_idx, doc in enumerate(documents):
            if doc is None:
                continue

            # TF Calculation
            term_counts = Counter(self._tokenize(doc))
            total_terms = sum(term_counts.values())

            if total_terms == 0: 
                continue

            for word, count in term_counts.items():
                if word in self.vocab:
                    col_idx = self.vocab[word]
                    # TF = count / doc_length
                    tf = count / total_terms
                    # TF-IDF = TF * IDF
                    tfidf_score = tf * self.idf_diag[col_idx]

                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(tfidf_score)

        # Return Sparse Matrix
        matrix = csr_matrix((data, (rows, cols)),
                           shape=(len(documents), self.vocab_size))
        return self._l2_normalize(matrix)

    def fit_transform(self, documents):
        """Optimization: Chain fit and transform."""
        return self.fit(documents).transform(documents)

    def _l2_normalize(self, matrix):
        """
        Apply L2 normalization to each row.
        Makes all document vectors unit length.
        """
        row_norms = np.sqrt(np.array(matrix.power(2).sum(axis=1)).flatten())
        row_norms[row_norms == 0] = 1  # Avoid division by zero

        norm_matrix = csr_matrix((1.0 / row_norms,
                                 (range(len(row_norms)), range(len(row_norms)))),
                                shape=(len(row_norms), len(row_norms)))
        return norm_matrix @ matrix


# ============================================================================
# COMPREHENSIVE TEST SUITE - Practice explaining these edge cases!
# ============================================================================

def run_comprehensive_tests():
    """Run all edge case tests."""
    print("=" * 70)
    print("TF-IDF VECTORIZER - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    # Test 1: Basic functionality
    print("\n[TEST 1] Basic Training & Transform")
    docs = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are friends"
    ]
    vectorizer = TFIDFVectorizer(min_df=1, max_df=3)
    X = vectorizer.fit_transform(docs)
    print(f"Vocab size: {vectorizer.vocab_size}")
    print(f"Matrix shape: {X.shape}")
    assert X.shape == (3, 12), f"Expected (3, 12), got {X.shape}"
    assert vectorizer.vocab_size == 12, f"Expected 12 terms, got {vectorizer.vocab_size}"
    print("✅ PASSED")

    # Test 2: Empty corpus handling
    print("\n[TEST 2] Empty Corpus")
    vectorizer_empty = TFIDFVectorizer()
    vectorizer_empty.fit([])
    X_empty = vectorizer_empty.transform([])
    assert X_empty.shape == (0, 0), "Empty corpus should return (0,0) matrix"
    assert vectorizer_empty.vocab_size == 0, "Empty corpus should have vocab_size=0"
    print("✅ PASSED")

    # Test 3: None input validation
    print("\n[TEST 3] None Input Validation")
    try:
        vectorizer_none = TFIDFVectorizer()
        vectorizer_none.fit(None)
        print("❌ FAILED - Should raise ValueError for None input")
        assert False
    except ValueError as e:
        print(f"✅ PASSED - Correctly raised ValueError: {e}")

    # Test 4: None documents in batch
    print("\n[TEST 4] None Documents in Batch")
    docs_with_none = ["first doc", None, "third doc"]
    vectorizer_mixed = TFIDFVectorizer()
    X_mixed = vectorizer_mixed.fit_transform(docs_with_none)
    assert X_mixed.shape[0] == 3, "Should handle None in batch"
    print(f"✅ PASSED - Shape: {X_mixed.shape}")

    # Test 5: Single document
    print("\n[TEST 5] Single Document")
    single_doc = ["hello world"]
    vectorizer_single = TFIDFVectorizer()
    X_single = vectorizer_single.fit_transform(single_doc)
    assert X_single.shape == (1, 2), f"Expected (1, 2), got {X_single.shape}"
    print("✅ PASSED")

    # Test 6: Empty documents (only spaces/punctuation)
    print("\n[TEST 6] Empty Documents (No Valid Tokens)")
    empty_docs = ["", "   ", "!!!", "###"]
    vectorizer_empty_docs = TFIDFVectorizer()
    X_empty_docs = vectorizer_empty_docs.fit_transform(empty_docs)
    assert vectorizer_empty_docs.vocab_size == 0, "No valid tokens should result in empty vocab"
    print("✅ PASSED")

    # Test 7: All words filtered by min_df
    print("\n[TEST 7] All Words Filtered by min_df")
    docs_rare = ["apple banana", "cherry date", "elderberry fig"]
    vectorizer_filtered = TFIDFVectorizer(min_df=2)  # Each word appears only once
    X_filtered = vectorizer_filtered.fit_transform(docs_rare)
    assert vectorizer_filtered.vocab_size == 0, "All rare words should be filtered"
    assert X_filtered.shape == (3, 0), f"Expected (3, 0), got {X_filtered.shape}"
    print("✅ PASSED")

    # Test 8: max_features limiting
    print("\n[TEST 8] max_features Limiting")
    docs_many = ["word" + str(i) for i in range(100)]  # 100 unique words
    docs_many = [" ".join(docs_many)]
    vectorizer_limited = TFIDFVectorizer(max_features=10)
    vectorizer_limited.fit(docs_many)
    assert vectorizer_limited.vocab_size == 10, f"Expected 10 features, got {vectorizer_limited.vocab_size}"
    print("✅ PASSED")

    # Test 9: max_df filtering
    print("\n[TEST 9] max_df Filtering (Common Words)")
    docs_common = ["the cat sat", "the dog sat", "the bird sat"]
    vectorizer_maxdf = TFIDFVectorizer(max_df=0.5)  # "the" and "sat" appear in all (100%)
    X_maxdf = vectorizer_maxdf.fit_transform(docs_common)
    # Only cat, dog, bird should remain (appear in 33% of docs each)
    assert vectorizer_maxdf.vocab_size == 3, f"Expected 3 words after max_df filter, got {vectorizer_maxdf.vocab_size}"
    print("✅ PASSED")

    # Test 10: Transform on new documents
    print("\n[TEST 10] Transform on New Documents")
    train_docs = ["apple orange", "banana grape"]
    test_docs = ["apple banana", "watermelon"]  # watermelon not in vocab
    vectorizer_new = TFIDFVectorizer()
    vectorizer_new.fit(train_docs)
    X_test = vectorizer_new.transform(test_docs)
    assert X_test.shape == (2, 4), f"Expected (2, 4), got {X_test.shape}"
    # watermelon should be ignored (not in vocab)
    print("✅ PASSED")

    # Test 11: L2 normalization verification
    print("\n[TEST 11] L2 Normalization")
    docs_norm = ["cat dog", "cat cat dog"]
    vectorizer_norm = TFIDFVectorizer()
    X_norm = vectorizer_norm.fit_transform(docs_norm)
    # Check that each row has L2 norm ≈ 1
    row_norms = np.sqrt(np.array(X_norm.power(2).sum(axis=1)).flatten())
    assert np.allclose(row_norms, 1.0, atol=1e-6), f"Rows not normalized: {row_norms}"
    print(f"✅ PASSED - Row norms: {row_norms}")

    # Test 12: Unicode handling
    print("\n[TEST 12] Unicode Characters")
    docs_unicode = ["café résumé", "naïve"]
    vectorizer_unicode = TFIDFVectorizer()
    X_unicode = vectorizer_unicode.fit_transform(docs_unicode)
    assert X_unicode.shape[0] == 2, "Should handle Unicode"
    print(f"✅ PASSED - Vocab: {list(vectorizer_unicode.vocab.keys())}")

    # Test 13: Case insensitivity
    print("\n[TEST 13] Case Insensitivity")
    docs_case = ["Apple BANANA", "apple banana"]
    vectorizer_case = TFIDFVectorizer()
    X_case = vectorizer_case.fit_transform(docs_case)
    assert vectorizer_case.vocab_size == 2, "Should treat Apple and apple as same"
    print("✅ PASSED")

    # Test 14: Single character words filtered
    print("\n[TEST 14] Single Character Words Filtered")
    docs_single = ["a b c apple banana"]
    vectorizer_single_char = TFIDFVectorizer()
    X_single_char = vectorizer_single_char.fit_transform(docs_single)
    assert "a" not in vectorizer_single_char.vocab, "Single chars should be filtered by regex"
    assert vectorizer_single_char.vocab_size == 2, "Only apple and banana should remain"
    print("✅ PASSED")

    # Test 15: Very long document
    print("\n[TEST 15] Very Long Document")
    long_doc = ["word " * 10000]
    vectorizer_long = TFIDFVectorizer()
    X_long = vectorizer_long.fit_transform(long_doc)
    assert X_long.shape == (1, 1), "Should handle long documents"
    print("✅ PASSED")

    print("\n" + "=" * 70)
    print("ALL 15 TESTS PASSED! 🎉")
    print("=" * 70)


# ============================================================================
# INTERVIEW TALKING POINTS - Memorize these!
# ============================================================================

def print_interview_points():
    print("\n" + "=" * 70)
    print("🎯 INTERVIEW TALKING POINTS")
    print("=" * 70)

    points = [
        ("Time Complexity",
         "fit(): O(N*M) where N=docs, M=avg_words. Single pass per doc.\n"
         "   transform(): O(N*M) for tokenization + O(N*M*log V) for vocab lookup.\n"
         "   Overall: O(N*M) assuming hash table O(1) lookup."),

        ("Space Complexity",
         "O(V) for vocabulary storage where V=unique terms.\n"
         "   O(N*V) worst case for dense matrix, but we use sparse CSR.\n"
         "   Sparse: O(nnz) where nnz=number of non-zero values."),

        ("Why Sparse Matrices?",
         "Text is naturally sparse: most words don't appear in most documents.\n"
         "   100K vocab × 10K docs = 1B entries, but maybe only 10M non-zero.\n"
         "   CSR format stores only (row, col, value) tuples → 100x memory savings."),

        ("IDF Formula",
         "idf = log((N+1)/(df+1)) + 1\n"
         "   +1 smoothing prevents division by zero.\n"
         "   +1 offset ensures positive IDF values (sklearn compatibility).\n"
         "   Higher IDF = rarer word = more discriminative."),

        ("L2 Normalization",
         "Divides each document vector by its Euclidean length.\n"
         "   Why? Prevents longer documents from dominating similarity.\n"
         "   'Cat dog' (2 words) vs 'cat dog cat dog...' (100 words) →\n"
         "   same direction, different magnitude.\n"
         "   After L2: both have length 1, only direction matters."),

        ("Edge Cases Handled",
         "1. None input → ValueError\n"
         "   2. Empty corpus → (0,0) matrix\n"
         "   3. None in batch → skip gracefully\n"
         "   4. Empty docs → skip (division by zero avoided)\n"
         "   5. All words filtered → empty vocab\n"
         "   6. Unknown words in transform → ignored"),

        ("Apple Use Case: App Store Search",
         "User query: 'photo editor with filters'\n"
         "   We vectorize query → TF-IDF vector\n"
         "   Compare with 2M app descriptions (pre-computed)\n"
         "   Use cosine similarity on L2-normalized vectors\n"
         "   Return top-k most similar apps\n"
         "   Sparse matrices → fits in memory, fast retrieval"),

        ("Production Improvements",
         "1. Streaming fit for large corpora (don't load all in memory)\n"
         "   2. Vocabulary pruning (remove rare/common terms)\n"
         "   3. N-grams (bigrams, trigrams) for better context\n"
         "   4. Sublinear TF scaling: 1 + log(tf) instead of raw tf\n"
         "   5. Model serialization (pickle/joblib)\n"
         "   6. Multilingual support (language-specific tokenization)\n"
         "   7. Incremental vocab updates (new terms over time)")
    ]

    for i, (title, explanation) in enumerate(points, 1):
        print(f"\n{i}. {title}")
        for line in explanation.split('\n'):
            print(f"   {line}")

    print("\n" + "=" * 70)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    run_comprehensive_tests()
    print_interview_points()

    # Quick demo
    print("\n" + "=" * 70)
    print("QUICK DEMO")
    print("=" * 70)

    docs = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are friends"
    ]

    vectorizer = TFIDFVectorizer(min_df=1, max_df=3)
    X = vectorizer.fit_transform(docs)

    print(f"\nVocabulary: {vectorizer.vocab}")
    print(f"IDF values: {vectorizer.idf_diag}")
    print(f"\nTF-IDF Matrix shape: {X.shape}")
    print(f"TF-IDF Matrix (dense view):")
    print(X.toarray())
