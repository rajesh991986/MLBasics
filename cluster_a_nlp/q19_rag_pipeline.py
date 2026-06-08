"""
Q19 — RAG Pipeline (Retrieval-Augmented Generation) [HIGH PROBABILITY]
Target time: 25 min | Requires: langchain, faiss-cpu, sentence-transformers

APPROACH (say this in the first 60 seconds):
"RAG has two phases. Offline: chunk documents into 300-500 token pieces,
embed each chunk, store in a vector database. Online: embed the user query,
ANN search retrieves the top-k most relevant chunks, inject them into the
LLM prompt as context, generate a grounded answer. Key guardrail: the prompt
says 'answer only from the context below' plus a faithfulness check scores
each claim against retrieved docs. If retrieval confidence is low, skip
the LLM entirely and escalate to a human."

WHY RAG > FINE-TUNING for knowledge retrieval:
  1. Freshness  — update the index, no retraining needed
  2. Explainability — you can show the user the source chunk
  3. Cost       — re-indexing is cheap; fine-tuning a 7B model is not
  Fine-tuning wins for domain-specific style/format, not knowledge.

CHUNKING RULES:
  Too small (<100 tokens)  → chunk lacks context to answer the question
  Too large (>1000 tokens) → irrelevant content dilutes retrieval signal
  Sweet spot: 300-500 tokens, 50-100 token overlap (boundary sentences intact)
  Structured docs (policies, FAQs) → semantic chunking at section boundaries

EVALUATION (three layers):
  Retrieval  : Recall@k — did the right chunk appear in top-k?
  Generation : Faithfulness (claims grounded in docs?), Answer Relevance
  End-to-end : Resolution rate, escalation rate, customer satisfaction

OLD IMPORTS (pre-0.2, will crash — shown for interview comparison):
  from langchain.vectorstores import FAISS          # ← removed
  from langchain.embeddings import HuggingFaceEmbeddings  # ← removed
  from langchain.llms import OpenAI                 # ← removed

MODERN IMPORTS (v0.2+, use these):
  from langchain_community.vectorstores import FAISS
  from langchain_huggingface import HuggingFaceEmbeddings
  from langchain_openai import ChatOpenAI
"""

# ─────────────────────────────────────────────────────────────
# Imports — modern LangChain (v0.2+)
# ─────────────────────────────────────────────────────────────
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ─────────────────────────────────────────────────────────────
# SECTION 1: Synthetic knowledge base (hotel/travel policies)
# No API key needed — we mock the LLM for demo purposes
# ─────────────────────────────────────────────────────────────

RAW_POLICY_DOCS = [
    """Cancellation Policy — Standard Rate
    Guests may cancel free of charge up to 24 hours before the scheduled
    check-in date. Cancellations made within 24 hours of check-in will be
    charged one night's stay as a cancellation fee. No-shows are charged
    the full booking amount. To cancel, log in to Manage My Booking or
    contact customer support.""",

    """Cancellation Policy — Non-Refundable Rate
    Non-refundable bookings cannot be cancelled or amended after confirmation.
    The full booking amount will be charged at the time of reservation.
    These rates are offered at a discount and do not qualify for refunds
    under any circumstances, including medical emergencies.""",

    """Cancellation Policy — Flexible Rate
    Fully flexible bookings may be cancelled at any time up to 6pm on the
    day of arrival at no charge. Cancellations after 6pm on the arrival day
    incur a one-night penalty fee. Changes to dates are permitted subject
    to availability and rate differences.""",

    """Refund Processing Times
    Refunds for eligible cancellations are processed within 5-10 business
    days to the original payment method. Credit card refunds may take an
    additional 3-5 days to appear depending on the issuing bank. Refunds
    for bookings paid with travel credits are returned as credits immediately.""",

    """Hotel Check-In and Check-Out
    Standard check-in time is 3:00 PM. Early check-in from 12:00 PM is
    available upon request, subject to availability, and may incur an
    additional fee. Check-out time is 11:00 AM. Late check-out until 2:00 PM
    can be requested at the front desk and is subject to availability.""",

    """Pet Policy
    Select properties are pet-friendly. A pet fee of $25-$50 per night may
    apply. Maximum two pets per room. Pets must be declared at booking.
    Service animals are always welcome at no additional charge as required
    by law. Please check the individual property listing for specific rules.""",
]


# ─────────────────────────────────────────────────────────────
# SECTION 2: Offline indexing pipeline
# ─────────────────────────────────────────────────────────────

def build_index(raw_texts: list, chunk_size: int = 400, chunk_overlap: int = 80):
    """
    Offline phase:
    1. Wrap raw strings as LangChain Documents
    2. Chunk with overlap so boundary sentences aren't cut off
    3. Embed each chunk with a sentence transformer
    4. Store in FAISS for fast ANN retrieval

    chunk_size and chunk_overlap are key hyperparameters:
      - chunk_size too small → chunks lack enough context
      - chunk_size too large → irrelevant content dilutes retrieval
      - overlap ensures sentences at boundaries appear in at least one chunk
    """
    # Step 1 — wrap as Documents
    documents = [
        Document(page_content=text, metadata={"source": f"policy_{i}"})
        for i, text in enumerate(raw_texts)
    ]

    # Step 2 — chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],  # prefer sentence boundaries
    )
    chunks = splitter.split_documents(documents)
    print(f"  {len(documents)} documents → {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")

    # Step 3+4 — embed and index
    # all-MiniLM-L6-v2: 384-dim, fast on CPU, strong semantic quality
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"  FAISS index built: {vectorstore.index.ntotal} vectors")
    return vectorstore, embeddings


# ─────────────────────────────────────────────────────────────
# SECTION 3: Online retrieval (no LLM needed to demo this)
# ─────────────────────────────────────────────────────────────

def retrieve(vectorstore, query: str, k: int = 3) -> list:
    """
    Online retrieval phase:
    1. Embed the query (same model as indexing — MUST match)
    2. ANN search → top-k chunks with similarity scores
    3. Return (chunk_text, score, metadata)

    Why same model matters: embedding spaces are model-specific.
    Mixing models = comparing vectors from different coordinate systems.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [(doc.page_content, float(score), doc.metadata) for doc, score in results]


def build_prompt(query: str, chunks: list) -> str:
    """
    Augmentation phase — build the grounded prompt.
    The 'answer only from context' instruction is the primary
    hallucination guardrail.
    """
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]: {text}" for i, (text, _, _) in enumerate(chunks)
    )
    return f"""You are a helpful travel assistant. Answer the user's question
using ONLY the context provided below. If the answer is not in the context,
say exactly: "I don't have that information. Please contact customer support."
Do not add information from your own knowledge.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


# ─────────────────────────────────────────────────────────────
# SECTION 4: Full RAG chain WITH a real LLM (optional)
# Requires: OPENAI_API_KEY env var
# If no key → demo runs retrieval only (most of the interview value)
# ─────────────────────────────────────────────────────────────

def build_rag_chain(vectorstore):
    """
    Modern LCEL (LangChain Expression Language) RAG chain — v0.2+ style.

    Pattern:  retriever | format_docs → prompt | llm | output_parser

    OLD pattern (pre-v0.2, now removed):
      RetrievalQA.from_chain_type(llm=..., chain_type="stuff", retriever=...)

    NEW pattern (LCEL, composable, streamable):
      chain = {"context": retriever | format_docs, "question": RunnablePassthrough()}
              | prompt | llm | StrOutputParser()

    chain_type equivalents in LCEL:
      "stuff"       → pass all chunks as one context string (default below)
      "map_reduce"  → run LLM per chunk, then combine — implement with batch()
      "refine"      → sequential refinement — implement with chained invoke()
    """
    try:
        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY not set")

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        def format_docs(docs):
            return "\n\n---\n\n".join(
                f"[Source {i+1}]: {d.page_content}"
                for i, d in enumerate(docs)
            )

        prompt = PromptTemplate.from_template(
            """You are a helpful travel assistant. Answer using ONLY the context below.
If the answer is not in the context, say: "I don't have that information."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        )

        # LCEL chain: dict feeds two keys, then prompt → llm → parse
        chain = (
            {"context": retriever | format_docs,
             "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    except (ImportError, EnvironmentError) as e:
        print(f"  [LLM skipped] {e}")
        print("  Running retrieval-only demo instead.")
        return None


# ─────────────────────────────────────────────────────────────
# SECTION 5: Faithfulness checker (hallucination detection)
# Aimé literally built this — he WILL ask about it
# ─────────────────────────────────────────────────────────────

import re
import math
from collections import Counter


def simple_faithfulness_score(answer: str, context_chunks: list) -> dict:
    """
    Lightweight faithfulness check: for each sentence in the answer,
    compute max cosine similarity against the retrieved chunks using
    token overlap (TF-IDF proxy — no model needed).

    Production approach: use an NLI model (e.g. cross-encoder/nli-deberta)
    to check entailment: does the context ENTAIL the claim?

    Scoring:
      >= 0.3 → grounded (claim has token overlap with a source)
      <  0.3 → potentially hallucinated
    """
    def token_vec(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return Counter(tokens)

    def cosine(a, b):
        shared = set(a) & set(b)
        if not shared:
            return 0.0
        dot = sum(a[k] * b[k] for k in shared)
        na = math.sqrt(sum(v**2 for v in a.values()))
        nb = math.sqrt(sum(v**2 for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    context_text = " ".join(c[0] for c in context_chunks)
    context_vec = token_vec(context_text)

    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 10]
    results = []
    for sentence in sentences:
        score = cosine(token_vec(sentence), context_vec)
        results.append({
            "claim": sentence[:60] + "...",
            "score": round(score, 3),
            "grounded": score >= 0.3,
        })

    overall = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {
        "overall_faithfulness": round(overall, 3),
        "verdict": "PASS" if overall >= 0.3 else "FAIL — possible hallucination",
        "claims": results,
    }


# ─────────────────────────────────────────────────────────────
# SECTION 6: Confidence-based routing
# ─────────────────────────────────────────────────────────────

def route_query(chunks_with_scores: list, low_conf_threshold: float = 0.8) -> str:
    """
    FAISS returns L2 distance (lower = more similar).
    If best score is above threshold → no good match found →
    skip LLM and escalate to human agent.

    In production: also check if query is out-of-domain
    (e.g. "what's the weather?" against a policy corpus).
    """
    if not chunks_with_scores:
        return "escalate"
    best_score = chunks_with_scores[0][1]   # L2 distance, lower is better
    return "llm" if best_score < low_conf_threshold else "escalate"


# ─────────────────────────────────────────────────────────────
# SECTION 7: Run everything
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SECTION 2: Build FAISS index")
    print("=" * 60)
    vectorstore, _ = build_index(RAW_POLICY_DOCS)

    test_queries = [
        "What is the cancellation policy for my hotel booking?",
        "How long does a refund take to process?",
        "Can I bring my dog to the hotel?",
        "What time is check-in?",
        "What happens if I booked a non-refundable rate?",
    ]

    print("\n" + "=" * 60)
    print("SECTION 3: Retrieval + Prompt + Faithfulness")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQ: {query}")

        # Retrieve
        chunks = retrieve(vectorstore, query, k=3)

        # Route
        routing = route_query(chunks, low_conf_threshold=0.8)
        print(f"  Routing: {routing}")

        if routing == "escalate":
            print("  → No confident match. Escalating to human agent.")
            continue

        # Show top retrieved chunk
        best_text, best_score, best_meta = chunks[0]
        print(f"  Top chunk (L2={best_score:.3f}, {best_meta}):")
        print(f"  '{best_text[:120].strip()}...'")

        # Build prompt (shown for interview — LLM call is optional)
        prompt = build_prompt(query, chunks)

        # Mock answer (simulating what LLM would return)
        mock_answer = best_text[:200].strip()

        # Faithfulness check
        faith = simple_faithfulness_score(mock_answer, chunks)
        print(f"  Faithfulness: {faith['overall_faithfulness']} — {faith['verdict']}")

    print("\n" + "=" * 60)
    print("SECTION 4: Full RAG chain (requires OPENAI_API_KEY)")
    print("=" * 60)
    chain = build_rag_chain(vectorstore)
    if chain:
        # LCEL chains take a plain string (the question), not a dict
        result = chain.invoke("What is the cancellation policy?")
        print("\nAnswer:", result)
    else:
        print("  Skipped — set OPENAI_API_KEY to run live generation.")

    print("\n" + "=" * 60)
    print("KEY IMPORT CHANGE (interview gotcha)")
    print("=" * 60)
    print("""
  OLD (pre-0.2, deprecated):              NEW (v0.2+):
  ────────────────────────────────────    ─────────────────────────────────────
  from langchain.vectorstores import      from langchain_community.vectorstores
    FAISS                                   import FAISS
  from langchain.embeddings import        from langchain_huggingface import
    HuggingFaceEmbeddings                   HuggingFaceEmbeddings
  from langchain.llms import OpenAI       from langchain_openai import
                                            ChatOpenAI
  from langchain.chat_models import       (same — ChatOpenAI moved to
    ChatOpenAI                              langchain_openai)
""")

    print("=" * 60)
    print("RAG EVALUATION METRICS")
    print("=" * 60)
    print("""
  Layer        Metric              What it measures
  ─────────────────────────────────────────────────────
  Retrieval    Recall@k            Right chunk in top-k?
  Retrieval    MRR                 Rank of first relevant chunk
  Generation   Faithfulness        Claims grounded in context?
  Generation   Answer Relevance    Does it answer the question?
  End-to-end   Resolution rate     User got their answer?
  End-to-end   Escalation rate     Fell back to human agent?
  End-to-end   CSAT                Customer satisfaction score
""")
    print("All sections complete.")
