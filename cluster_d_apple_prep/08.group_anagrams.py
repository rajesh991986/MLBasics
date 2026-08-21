"""
Group Anagrams — LC 49

APPROACH: Sort each string → canonical key → group by key
  Anagrams share the same sorted characters. Use tuple(sorted(s)) as dict key.

COMPLEXITY:
  Time:  O(n × k log k)  — sorting each of n strings of length k
  Space: O(n × k)        — storing all strings in groups

APPLE USE CASE:
  Deduplicating search queries where word order varies.
  "return item" and "item return" → same intent group.
"""

from collections import defaultdict


def group_anagrams(strs):
    """
    Args:
        strs (list): List of strings.
    Returns:
        list of lists: Groups of anagrams.
    """
    groups = defaultdict(list)

    for s in strs:
        key = tuple(sorted(s))     # "eat" → ('a','e','t')
        groups[key].append(s)

    return list(groups.values())


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("GROUP ANAGRAMS TESTS")
print("=" * 60)

# Test 1: Standard case
result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
result_sets = [set(g) for g in result]
assert {"eat", "tea", "ate"} in result_sets
assert {"tan", "nat"} in result_sets
assert {"bat"} in result_sets
print(f"\n[Test 1] Standard → {len(result)} groups: {[sorted(g) for g in result]}")
print("✅ Test 1 PASSED")

# Test 2: Single empty string
result = group_anagrams([""])
assert result == [[""]]
print(f"\n[Test 2] [\"\"] → {result}")
print("✅ Test 2 PASSED")

# Test 3: Single character
result = group_anagrams(["a"])
assert result == [["a"]]
print(f"\n[Test 3] [\"a\"] → {result}")
print("✅ Test 3 PASSED")

# Test 4: All same group
result = group_anagrams(["abc", "bca", "cab"])
assert len(result) == 1 and len(result[0]) == 3
print(f"\n[Test 4] All anagrams → 1 group of 3")
print("✅ Test 4 PASSED")

# Test 5: No anagrams — each in own group
result = group_anagrams(["abc", "def", "ghi"])
assert len(result) == 3
print(f"\n[Test 5] No anagrams → {len(result)} groups (one each)")
print("✅ Test 5 PASSED")

print("\n" + "=" * 60)
print("ALL 5 TESTS PASSED ✅")
print("=" * 60)
