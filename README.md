# MLBasics

A practical ML interview prep repo — 37 Python files across two tiers:

- **From-scratch implementations** — internals, math, complexity (understand *how* algorithms work)
- **Applied interview patterns** — pandas wrangling, SQL, end-to-end models, RAG (what live coding screens actually test)

Every file runs standalone. Each applied file has a **"say this first"** verbal intro, inline comments on every non-obvious line explaining *why not just what*, and verified output.

---

## Quick Start

```bash
git clone git@github.com:rajeshdnp/MLBasics.git
cd MLBasics

pip install numpy pandas scikit-learn torch lightgbm sentence-transformers \
            langchain langchain-community langchain-huggingface \
            langchain-text-splitters langchain-openai faiss-cpu scipy matplotlib

# Run any file — each is self-contained
python3 cluster_b_ml/q27_end_to_end_model.py
python3 cluster_b_ml/q26_sql_patterns.py
python3 cluster_a_nlp/q18_tfidf_semantic_similarity.py
```

---

## Interview Priority Order

If you have limited time, read in this order — mapped to what live screens actually ask:

| Priority | File | Why it matters |
|---|---|---|
| ⭐⭐⭐ | `q25_pandas_wrangling` | Near-certain warm-up at every ML screen |
| ⭐⭐⭐ | `q26_sql_patterns` | CASE + window functions — reported live task |
| ⭐⭐⭐ | `q27_end_to_end_model` | Full pipeline: EDA → LR baseline → LightGBM → threshold tuning |
| ⭐⭐⭐ | `q18_tfidf_semantic_similarity` | "Compute text similarity" — most-reported NLP coding task |
| ⭐⭐⭐ | `q19_rag_pipeline` | RAG + FAISS — the hard variant; NLP interviewers probe this |
| ⭐⭐ | `q21_recommender_system` | Two-stage recsys + NDCG — core to ranking-focused teams |
| ⭐⭐ | `q22_nn_price_estimation` | Regression NN — reported live CodePair task |
| ⭐⭐ | `q23_bias_variance_regularization` | Guaranteed theory question at every ML loop |
| ⭐⭐ | `q24_ab_testing` | Classic "how would you A/B test X?" — Expedia staple |
| ⭐⭐ | `q28_from_scratch_functions` | Warm-up: RMSE, variance, P/R/F1, softmax, cosine sim |
| ⭐ | `q10_metrics` | Ranking metrics: P@k, MRR, NDCG from scratch |
| ⭐ | `q3_cosine_similarity` | Sparse cosine, document ranking from scratch |
| ⭐ | `q7_llm_evaluation` | Faithfulness scoring — ties directly to RAG |
| ⭐ | `q8_bm25_retriever` | BM25 from scratch — retrieval baseline |
| ⭐ | `q6_logistic_regression` | Gradient descent, L2 regularisation from scratch |

> **One hour before a screen:** `q25` → `q26` → `q27` → `q18`. That's the highest-yield core.

---

## File Map

### Cluster A — NLP

| File | What it covers | Tier |
|---|---|---|
| `q1_ngram_predictor.py` | N-gram LM, Laplace smoothing, backoff | From scratch |
| `q3_cosine_similarity.py` | Sparse cosine similarity, document ranking | From scratch |
| `q7_llm_evaluation.py` | Faithfulness scoring, claim grounding, RAG eval | From scratch |
| `q8_bm25_retriever.py` | Inverted index, BM25, IDF, TF saturation | From scratch |
| `q9_chunking_embedding.py` | Sentence chunking, vector store, retrieval | From scratch |
| `q10_metrics.py` | P/R/F1, P@k, MRR, NDCG | From scratch |
| `q17_feature_extraction.py` | Regex NER, keyword extraction, entity detection | From scratch |
| `q18_tfidf_semantic_similarity.py` | TF-IDF joint fit, sentence embeddings, ANN scale-up (FAISS) | **Applied** |
| `q19_rag_pipeline.py` | FAISS indexing, retrieval, faithfulness check, LCEL chain | **Applied** |

### Cluster B — ML Algorithms

| File | What it covers | Tier |
|---|---|---|
| `q2_naive_bayes.py` | Naive Bayes text classifier, log-probabilities | From scratch |
| `q4_mlp_backprop.py` | MLP forward + backprop (NumPy), chain rule, He init | From scratch |
| `q4_1_mlp_backprop_pytorch.py` | PyTorch MLP, autograd, CrossEntropyLoss | From scratch |
| `q4_2_mlp_backprop_pytorch.py` | PyTorch MLP multi-class, full training loop | From scratch |
| `q5_kmeans.py` | K-means, centroid update, convergence | From scratch |
| `q6_logistic_regression.py` | Logistic regression, gradient descent, L2 regularisation | From scratch |
| `q12_softmax_crossentropy.py` | Softmax, cross-entropy, numerical stability | From scratch |
| `q15_knn.py` | KNN, distance metrics, majority vote, tie-breaking | From scratch |
| `q20_price_tier_prediction.py` | Full ML pipeline: LR vs XGBoost, drift detection, monitoring | **Applied** |
| `q21_recommender_system.py` | User-based CF + content-based + hybrid recsys, NDCG@k | **Applied** |
| `q22_nn_price_estimation.py` | Regression NN: log-transform target, BatchNorm, Dropout, early stopping | **Applied** |
| `q23_bias_variance_regularization.py` | Bias-variance via polynomial degree, L1 vs L2 vs ElasticNet, dropout | **Applied** |
| `q24_ab_testing.py` | Power analysis, z-test, guardrail metrics, 6 pitfalls, Bayesian bonus | **Applied** |
| `q25_pandas_wrangling.py` | Load → inspect → clean → text clean → groupby → merge → encode | **Applied** |
| `q26_sql_patterns.py` | CASE, window functions (ROW_NUMBER/LAG/NTILE), CTEs, SQL↔pandas cheat sheet | **Applied** |
| `q27_end_to_end_model.py` | EDA → feature engineering → LR baseline → LightGBM → threshold tuning → test eval | **Applied** |
| `q28_from_scratch_functions.py` | RMSE/MAE/MAPE/R², sample variance, P/R/F1, sigmoid/softmax/ReLU, weighted sampler | From scratch |

### Cluster C — Neural Network Blocks + DP

| File | What it covers | Tier |
|---|---|---|
| `q11_layer_norm.py` | Layer normalisation, gamma/beta learnable params | From scratch |
| `q13_attention.py` | Scaled dot-product + multi-head attention, causal mask | From scratch |
| `q13_kvcache.py` | KV cache mechanics — how inference reuse works | From scratch |
| `q14_word2vec.py` | Word2Vec skip-gram, negative sampling, embeddings | From scratch |
| `q16_edit_distance.py` | Levenshtein DP, backtracking, space optimisation | From scratch |
| `q17_feature_extraction_bio.py` | Bio NER feature extraction | From scratch |
| `bpe.py` | Byte-pair encoding tokeniser | From scratch |

### Basics — Fundamentals

| File | What it covers |
|---|---|
| `01.tfidf.py` | TF-IDF vectors from scratch |
| `02.preprocess.py` | Text preprocessing pipeline |
| `03.text_classification_pipeline.py` | End-to-end text classification |
| `04.naive_bayes.py` | Naive Bayes classifier |
| `05.logisticregression.py` | Logistic regression |
| `06.kmeans.py` | K-means clustering |
| `06.minibatchkmeans.py` | Mini-batch K-means |
| `07.confusion_matrix_from_scratch.py` | Confusion matrix from scratch |
| `08.precision_recall_f1_from_scratch.py` | P/R/F1 from scratch |
| `09.cosine_similarity_from_scratch.py` | Cosine similarity from scratch |
| `010.edit_distance_from_scratch.py` | Edit distance (DP) |
| `011. ProductionTextClassifier.py` | Production-ready text classifier |

---

## Applied File Structure

Every applied file (`q18`–`q28`) follows the same pattern so you can scan fast:

```
DOCSTRING
  SAY THIS FIRST  — verbal intro to say before touching the keyboard (30–60 sec)
  KEY DECISIONS   — justify each design choice: metric, model, feature, threshold
  TALKING POINTS  — what the interviewer will probe on

══ SECTION N: NAME ══════════════════════════════
  # comment: what this line does
  code()                        # comment: why this choice was made
```

---

## Patterns to Know Cold

### Pandas muscle memory
```python
df.shape, df.dtypes, df.isnull().sum(), df.describe()
df['col'].fillna(df['col'].median())          # median, not mean — robust to outliers
df.groupby('city').agg(rev=('price','sum'), cnt=('id','count'))
df[(df['a'] > 0) & (df['b'] == 1)]           # compound filter — & not 'and', () required
df.merge(other, on='key', how='left')         # left = keep all rows from left table
pd.get_dummies(df, columns=['cat'], drop_first=False)  # False = better for tree models
df.groupby('city')['price'].rank(pct=True)    # percentile rank within group
```

### SQL patterns
```sql
-- CASE: if/elif/else in SQL — evaluated row by row
CASE WHEN price < 150 THEN 'budget' WHEN price < 250 THEN 'mid' ELSE 'premium' END

-- Window function: operates across rows without collapsing them (unlike GROUP BY)
ROW_NUMBER() OVER (PARTITION BY city ORDER BY price DESC)  -- rank within group
LAG(price, 1) OVER (PARTITION BY user_id ORDER BY date)    -- previous row's value
SUM(rev) OVER (PARTITION BY user ORDER BY date ROWS UNBOUNDED PRECEDING)  -- cumulative

-- CTE: named temp table — cleaner than nested subqueries
WITH top_per_user AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY rev DESC) AS rn FROM t
)
SELECT * FROM top_per_user WHERE rn = 1
```

### Model pipeline
```python
# STEP 0: check class balance → pick metric BEFORE picking model
#         imbalanced → PR-AUC primary, ROC-AUC secondary (never raw accuracy)

# STEP 1: feature engineering
#         log1p(skewed), one-hot(categoricals), interaction terms

# STEP 2: stratified 70/15/15 split (stratify preserves class ratio)
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.176, stratify=y_tv)

# STEP 3: LR baseline (interpretable, fast — if LR is good, stop here)
Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(class_weight='balanced'))])

# STEP 4: LightGBM (non-linear, handles categoricals, scale_pos_weight for imbalance)
LGBMClassifier(scale_pos_weight=max(1, neg/pos), ...)

# STEP 5: threshold tuning (0.5 is rarely optimal for imbalanced data)
precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
best_thresh = thresholds[np.argmax(2*P*R/(P+R))]

# STEP 6: test set ONCE at the very end — touching earlier = leakage
```

### Metric cheat sheet

| Metric | Formula | When to use |
|---|---|---|
| RMSE | √mean((y-ŷ)²) | Regression, penalises large errors |
| MAE | mean(\|y-ŷ\|) | Regression, robust to outliers |
| MAPE | mean(\|y-ŷ\|/\|y\|)×100 | Regression, % interpretation for stakeholders |
| PR-AUC | area under precision-recall curve | Imbalanced classification (primary) |
| ROC-AUC | area under ROC curve | Classification (secondary) |
| NDCG@k | graded relevance + position discount | Ranking (hotel/flight search) |
| F1 | 2PR/(P+R) | When both precision and recall matter |

---

## Requirements

```
Python 3.9+  (tested on 3.12)

pip install numpy pandas scikit-learn torch lightgbm scipy matplotlib \
            sentence-transformers langchain langchain-community \
            langchain-huggingface langchain-text-splitters \
            langchain-openai faiss-cpu
```

Most from-scratch files (`q1`–`q17`, `cluster_c`) need only `numpy` + stdlib.
Applied files (`q18`–`q28`) need the full dependencies above.
