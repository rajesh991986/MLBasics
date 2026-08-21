"""
Minimum Window Substring — LC 76

APPROACH: Sliding window with need Counter and missing count
  Expand right to satisfy t. Shrink left while still valid, record smallest window.
  'missing' tracks total chars still needed (not unique chars).

COMPLEXITY:
  Time:  O(|s| + |t|)  — each pointer moves at most |s| times
  Space: O(|t|)         — Counter

KEY INSIGHT:
  need[char] can go negative (excess of that char in window).
  need[char] > 0 means we STILL need more of that char.
  need[char] ≤ 0 means we have enough (or too many).
  When shrinking: only increment missing if giving back char that was needed.
"""

from collections import Counter


def min_window(s, t):
    """
    Args:
        s (str): Source string to search in.
        t (str): Target string — find smallest window containing all chars of t.
    Returns:
        str: Minimum window substring, or "" if not found.
    """
    if not t or not s:
        return ""

    need    = Counter(t)
    missing = len(t)     # total chars still needed
    left    = 0
    result  = ""

    for right, char in enumerate(s):
        if need[char] > 0:
            missing -= 1
        need[char] -= 1

        while missing == 0:
            window = s[left:right + 1]
            if not result or len(window) < len(result):
                result = window
            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1

    return result


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("MINIMUM WINDOW SUBSTRING TESTS")
print("=" * 60)

# Test 1: Standard case
result = min_window("ADOBECODEBANC", "ABC")
assert result == "BANC", f"Expected 'BANC', got '{result}'"
print(f"\n[Test 1] 'ADOBECODEBANC' target='ABC' → '{result}'")
print("✅ Test 1 PASSED")

# Test 2: Exact match
result = min_window("a", "a")
assert result == "a"
print(f"\n[Test 2] 'a' target='a' → '{result}'")
print("✅ Test 2 PASSED")

# Test 3: Impossible — t longer than s
result = min_window("a", "aa")
assert result == ""
print(f"\n[Test 3] 'a' target='aa' → '{result}'  (impossible)")
print("✅ Test 3 PASSED")

# Test 4: t has duplicates
result = min_window("aa", "aa")
assert result == "aa"
print(f"\n[Test 4] 'aa' target='aa' → '{result}'")
print("✅ Test 4 PASSED")

# Test 5: Empty string
result = min_window("", "a")
assert result == ""
print(f"\n[Test 5] '' target='a' → '{result}'")
print("✅ Test 5 PASSED")

# Test 6: Window at start
result = min_window("abc", "ab")
assert result == "ab"
print(f"\n[Test 6] 'abc' target='ab' → '{result}'")
print("✅ Test 6 PASSED")

print("\n" + "=" * 60)
print("ALL 6 TESTS PASSED ✅")
print("=" * 60)
