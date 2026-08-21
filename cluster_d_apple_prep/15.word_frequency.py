"""
Word Frequency Count

APPROACH: regex tokenize + Counter
  re.findall(r'\b[a-zA-Z]+\b') handles punctuation cleanly.
  .lower() for case-insensitive counting.

COMPLEXITY:
  Time:  O(n)  — single pass through text
  Space: O(v)  — v unique words in vocabulary

APPLE USE CASE:
  Analyze most common user queries to Siri to identify top intent categories.
  "What do users ask most?" → word frequency on query logs.

EDGE CASES:
  Punctuation attached: "hello," → regex strips it cleanly.
  Unicode: é, ü etc. — \b[a-zA-Z]+ only matches ASCII letters.
  Hyphenated: "well-known" → two tokens "well" and "known".
  Ask interviewer: "treat as one word or two?"
"""

from collections import Counter
import re


def word_frequency(text):
    """
    Args:
        text (str): Input text.
    Returns:
        dict: word → count mapping.
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return dict(Counter(words))


def top_k_words(text, k):
    """Return top K most frequent words as [(word, count), ...]."""
    freq = Counter(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
    return freq.most_common(k)


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("WORD FREQUENCY TESTS")
print("=" * 60)

# Test 1: Standard case
text = "the quick brown fox jumps over the lazy dog the fox"
result = word_frequency(text)
assert result["the"] == 3
assert result["fox"] == 2
print(f"\n[Test 1] 'the' → {result['the']}, 'fox' → {result['fox']}")
print("✅ Test 1 PASSED")

# Test 2: Punctuation stripped
result = word_frequency("Hello, world! Hello.")
assert result["hello"] == 2 and result["world"] == 1
print(f"\n[Test 2] 'Hello,' and 'Hello.' both count as 'hello' → {result['hello']}")
print("✅ Test 2 PASSED")

# Test 3: Case insensitive
result = word_frequency("Apple apple APPLE")
assert result["apple"] == 3
print(f"\n[Test 3] 'Apple', 'apple', 'APPLE' → {result['apple']}")
print("✅ Test 3 PASSED")

# Test 4: Empty string
result = word_frequency("")
assert result == {}
print(f"\n[Test 4] Empty string → {result}")
print("✅ Test 4 PASSED")

# Test 5: Top K words
result = top_k_words("siri siri siri apple apple google", 2)
assert result[0] == ("siri", 3)
assert result[1] == ("apple", 2)
print(f"\n[Test 5] Top 2 words → {result}")
print("✅ Test 5 PASSED")

# Test 6: Numbers ignored (only letters matched)
result = word_frequency("order 12345 shipped")
assert "12345" not in result
assert result.get("order") == 1
print(f"\n[Test 6] Numbers ignored → {result}")
print("✅ Test 6 PASSED")

print("\n" + "=" * 60)
print("ALL 6 TESTS PASSED ✅")
print("=" * 60)
