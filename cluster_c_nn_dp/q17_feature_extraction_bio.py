"""
Q17 — Feature Extraction (Interview-Speed Version)
Apple Sr. Applied ML Interview | CoderPad-safe (stdlib only)

APPROACH (say this in first 60 seconds):
  "Part A: I'll keep a list of (label, regex) pairs in priority order — DATE first
   because it's most specific. I collect all matches, sort by priority then position,
   then do a greedy non-overlapping selection.
   Part B: single-pass state machine — two states, IDLE and IN_ENTITY. flush()
   commits the buffer. Three edge cases: malformed I- with no open entity gets
   discarded; mismatched type flushes and skips; end of sequence triggers a final
   flush so we don't drop the last entity."

COMPLEXITY: Time O(P*N + M^2) where P=patterns, N=text len, M=matches | Space O(M)
"""

import re

# ─── PART A ───────────────────────────────────────────────
# Returns: list of (text, label, start, end)

PATTERNS = [
    ("DATE",      re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+\d{4}\b", re.I)),
    ("AMOUNT",    re.compile(r"[\$€£¥]?\d[\d,]*(?:\.\d{2})?\s*(?:USD|EUR|GBP|million|M|B)?\b", re.I)),
    ("TERRITORY", re.compile(r"\b(?:United States|United Kingdom|European Union|Worldwide|Global|APAC|EMEA)\b", re.I)),
    ("PARTY",     re.compile(r"\b[A-Z][a-zA-Z\s]+(?:Inc\.|LLC|Ltd\.?|Corp\.?)\b")),
]

def extract_entities(text, patterns=None):
    if not text:
        return []
    patterns = patterns or PATTERNS

    # collect all candidates
    candidates = []
    for label, regex in patterns:
        for m in regex.finditer(text):
            candidates.append((m.start(), m.end(), label, m.group()))

    # sort: pattern order (priority) first, then start position
    order = {label: i for i, (label, _) in enumerate(patterns)}
    candidates.sort(key=lambda c: (order[c[2]], c[0]))

    # greedy non-overlapping selection
    taken = []
    results = []
    for start, end, label, matched in candidates:
        if not any(s < end and start < e for s, e in taken):
            taken.append((start, end))
            results.append((matched, label, start, end))

    results.sort(key=lambda x: x[2])   # sort by start offset
    return results


# ─── PART B ───────────────────────────────────────────────
# Returns: list of (text, label, start_token, end_token)

def decode_bio(tokens, tags):
    if len(tokens) != len(tags):
        raise ValueError("tokens and tags must be the same length")

    results = []
    buf, label, start = [], None, 0

    def flush(end):
        nonlocal buf, label
        if label and buf:
            results.append((" ".join(buf), label, start, end))
        buf, label = [], None

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag == "O":
            flush(i)
        elif tag.startswith("B-"):
            flush(i)
            label, buf, start = tag[2:], [token], i
        elif tag.startswith("I-"):
            if label is None or tag[2:] != label:  # malformed or mismatch
                flush(i)
            else:
                buf.append(token)
        else:
            flush(i)   # unknown tag → treat as O

    flush(len(tokens))
    return results


# ─── smoke test ───────────────────────────────────────────
if __name__ == "__main__":
    text = ("Apple Inc. and Acme Ltd. agree on January 15, 2024 "
            "for distribution in the United States. Fee: $2,500,000 USD.")

    print("Part A:")
    for ent in extract_entities(text):
        print(f"  {ent}")

    print("\nPart B:")
    tokens = ["Apple", "Inc", "signed", "with", "Acme", "Ltd", "on", "2024-01-15"]
    tags   = ["B-PARTY","I-PARTY","O","O","B-PARTY","I-PARTY","O","B-DATE"]
    for ent in decode_bio(tokens, tags):
        print(f"  {ent}")

    # edge cases
    print("\n  malformed I-:", decode_bio(["Inc"], ["I-PARTY"]))
    print("  mismatch:    ", decode_bio(["Apple","2024"],["B-PARTY","I-DATE"]))
    print("  empty:       ", decode_bio([], []))
