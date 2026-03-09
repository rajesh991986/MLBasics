import re
from collections import Counter, defaultdict

# ─────────────────────────────────────────────
# HELPER: Build initial character-level vocabulary
# ─────────────────────────────────────────────
def build_vocab(corpus: list[str]) -> dict:
    """
    Split every word into characters + end-of-word marker </w>.
    
    'low' → ('l', 'o', 'w', '</w>')
    
    Returns dict: {token_tuple: frequency}
    """
    vocab = Counter()
    for text in corpus:
        for word in text.split():
            # tuple of chars + end-of-word sentinel
            char_tuple = tuple(list(word) + ['</w>'])
            vocab[char_tuple] += 1
    return vocab


# ─────────────────────────────────────────────
# HELPER: Count all adjacent pairs across vocab
# ─────────────────────────────────────────────
def get_pair_counts(vocab: dict) -> Counter:
    """
    For each word-tuple, slide a window of 2 and count bigrams.
    Weight by word frequency.
    
    ('l','o','w','</w>') with freq=5 → 
        ('l','o'): 5, ('o','w'): 5, ('w','</w>'): 5
    """
    pairs = Counter()
    for token_tuple, freq in vocab.items():
        for i in range(len(token_tuple) - 1):
            pairs[(token_tuple[i], token_tuple[i+1])] += freq
    return pairs


# ─────────────────────────────────────────────
# HELPER: Merge a specific pair everywhere in vocab
# ─────────────────────────────────────────────
def merge_pair(pair: tuple, vocab: dict) -> dict:
    """
    Replace every occurrence of pair ('l','o') with merged 'lo'
    across all token tuples.
    """
    new_vocab = {}
    bigram = ' '.join(pair)          # 'l o'  — for regex replacement
    replacement = ''.join(pair)      # 'lo'

    for token_tuple, freq in vocab.items():
        # join to string, replace, split back to tuple
        token_str = ' '.join(token_tuple)
        merged_str = token_str.replace(bigram, replacement)
        new_token_tuple = tuple(merged_str.split())
        new_vocab[new_token_tuple] = freq

    return new_vocab


# ─────────────────────────────────────────────
# MAIN: Train BPE
# ─────────────────────────────────────────────
def train_bpe(corpus: list, num_merges: int) -> tuple:
    """Returns (merge_rules, vocabulary)."""
    vocab = build_vocab(corpus)
    merges = []          # ordered list of (pair → merged_token)

    for step in range(num_merges):
        pair_counts = get_pair_counts(vocab)
        
        if not pair_counts:
            break        # Edge case: no more pairs to merge
        
        # Greedy: always merge the MOST FREQUENT pair
        best_pair = max(pair_counts, key=pair_counts.get)
        merged_token = ''.join(best_pair)
        
        vocab = merge_pair(best_pair, vocab)
        merges.append((best_pair, merged_token))

        print(f"Step {step+1}: merge {best_pair} → '{merged_token}' "
              f"(count={pair_counts[best_pair]})")

    # Extract final vocab set
    final_vocab = set()
    for token_tuple in vocab:
        for token in token_tuple:
            final_vocab.add(token)

    return merges, final_vocab


# ─────────────────────────────────────────────
# APPLY: Tokenize new text with learned merges
# ─────────────────────────────────────────────
def tokenize_bpe(text: str, merges: list) -> list:
    """
    Apply merge rules in LEARNED ORDER to a new string.
    Order matters — merges applied out of order give wrong results.
    """
    tokens = []
    for word in text.split():
        # Start as characters + end marker
        word_tokens = list(word) + ['</w>']

        # Apply each merge rule in order
        for (pair, merged) in merges:
            i = 0
            new_tokens = []
            while i < len(word_tokens):
                # Check if current+next == pair to merge
                if (i < len(word_tokens) - 1 and
                        word_tokens[i] == pair[0] and
                        word_tokens[i+1] == pair[1]):
                    new_tokens.append(merged)
                    i += 2          # skip both tokens in pair
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            word_tokens = new_tokens

        tokens.extend(word_tokens)

    return tokens


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    corpus = ["low low low low low",
              "lower lower lower",
              "newest newest newest newest",
              "widest widest"]

    print("=== Training BPE (10 merges) ===")
    merges, vocab = train_bpe(corpus, num_merges=10)

    print(f"\nFinal vocab ({len(vocab)} tokens): {sorted(vocab)}")

    print("\n=== Tokenizing new text ===")
    test_words = ["low", "lower", "newest", "lowest", "unknown"]
    for word in test_words:
        result = tokenize_bpe(word, merges)
        print(f"  '{word}' → {result}")

"""
## 🧠 What the Output Teaches You

Running on the classic BPE corpus, the first few merges should be:
```
Step 1: merge ('e', 's') → 'es'     # 'newest', 'widest' both have 'es'
Step 2: merge ('es', 't') → 'est'   # extends the merge
Step 3: merge ('est', '</w>') → 'est</w>'
Step 4: merge ('l', 'o') → 'lo'     # 'low', 'lower'
Step 5: merge ('lo', 'w') → 'low'
...

"""
"""
this is this


vocab : 
(t,h,i,s,/w) = 2
    
(i,s,/w) = 1

Pair counts : 
    -> (i,s),(s,/w)

-> (t,h)=2,(h,i)=2,(i,s)=3,(s,/w)=3

top is (i,s)=3,(s,/w)=3

Merge : 
lets merge i,s

(i,s) -> (i s) , is

(t,h,i,s,/w) = 2
    -> (t h i s /w) -> replace i s with is -> t h is /w
    -> split again 
    ->(t,h) = 2,(h,is)=2, (is /w) =2 
    
(i,s,/w) = 1
    -> "i s /w" -> replace i s with is -> "is /w"
    -> "is /w".split()-> [is,/w]
    -> tuples -> (is,/w) +=1 = 3

(t,h) = 2,(h,is)=2, (is /w) =3
repeat 
pick (is,/w)=3

vocab : "t h is /w","is /w"

replace is,/w with is/w

"""