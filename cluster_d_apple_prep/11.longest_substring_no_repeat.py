"""
Longest Substring Without Repeating Characters — LC 3

APPROACH: Sliding window + last-seen hashmap
  Expand right pointer. On duplicate, move left past previous occurrence.
  Track last seen INDEX (not just presence) to handle out-of-window chars.

COMPLEXITY:
  Time:  O(n)  — each character processed once
  Space: O(min(n, charset))  — hashmap

CRITICAL EDGE CASE:
  "abba" — when we see 'a' at index 3, last_seen['a']=0 but left=2.
  The 'a' is NOT in current window, so we must NOT move left back.
  Guard: only update left if last_seen[char] >= left.
"""


def length_of_longest_substring(s):
    """
    Args:
        s (str): Input string.
    Returns:
        int: Length of longest substring without repeating characters.
    """
    last_seen = {}   # char → last seen index
    left      = 0
    max_len   = 0

    for right, char in enumerate(s):
        if char in last_seen and last_seen[char] >= left:
            left = last_seen[char] + 1      # shrink window from left

        last_seen[char] = right
        max_len = max(max_len, right - left + 1)

    return max_len


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("LONGEST SUBSTRING NO REPEAT TESTS")
print("=" * 60)

# Test 1: Standard case
result = length_of_longest_substring("abcabcbb")
assert result == 3, f"Expected 3, got {result}"
print(f"\n[Test 1] 'abcabcbb' → {result}  ('abc')")
print("✅ Test 1 PASSED")

# Test 2: All same character
result = length_of_longest_substring("bbbbb")
assert result == 1
print(f"\n[Test 2] 'bbbbb' → {result}")
print("✅ Test 2 PASSED")

# Test 3: Answer at end
result = length_of_longest_substring("pwwkew")
assert result == 3
print(f"\n[Test 3] 'pwwkew' → {result}  ('wke')")
print("✅ Test 3 PASSED")

# Test 4: Empty string
result = length_of_longest_substring("")
assert result == 0
print(f"\n[Test 4] '' → {result}")
print("✅ Test 4 PASSED")

# Test 5: Critical edge case — 'abba'
result = length_of_longest_substring("abba")
assert result == 2, f"Expected 2 (not 3), got {result}"
# At 'a' (index 3): last_seen['a']=0 but left=2. 0 < 2 so DON'T move left back.
# Window is "ba" (left=2 to right=3) → length 2.
print(f"\n[Test 5] 'abba' → {result}  (critical: 'a' at idx 3, last_seen=0 < left=2, no move)")
print("✅ Test 5 PASSED")

# Test 6: Single character
result = length_of_longest_substring("a")
assert result == 1
print(f"\n[Test 6] 'a' → {result}")
print("✅ Test 6 PASSED")

print("\n" + "=" * 60)
print("ALL 6 TESTS PASSED ✅")
print("=" * 60)
