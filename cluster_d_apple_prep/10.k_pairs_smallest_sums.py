"""
Find K Pairs with Smallest Sums — LC 373

APPROACH: Min-heap seeded with first row
  Both arrays sorted. Smallest sum starts at (nums1[0], nums2[0]).
  Seed heap with (nums1[0], nums2[j]) for j in range(min(k, len(nums2))).
  When popping (i, j): push (i+1, j) — advance in nums1, same j column.

COMPLEXITY:
  Time:  O(k log k)  — at most k heap operations
  Space: O(k)

KEY INSIGHT:
  We seed only the first row (i=0), not first column (j=0).
  Each j column is independent — no duplicate pairs this way.
"""

import heapq


def k_smallest_pairs(nums1, nums2, k):
    """
    Args:
        nums1 (list): First sorted array.
        nums2 (list): Second sorted array.
        k (int): Number of pairs to return.
    Returns:
        list of [a, b]: K pairs with smallest a+b.
    """
    if not nums1 or not nums2:
        return []

    # Seed with first element of nums1 paired with each of nums2[:k]
    heap = [(nums1[0] + nums2[j], 0, j)
            for j in range(min(k, len(nums2)))]
    heapq.heapify(heap)

    result = []
    while heap and len(result) < k:
        total, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])
        if i + 1 < len(nums1):
            heapq.heappush(heap, (nums1[i+1] + nums2[j], i+1, j))

    return result


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("K PAIRS SMALLEST SUMS TESTS")
print("=" * 60)

# Test 1: Standard case
result = k_smallest_pairs([1, 7, 11], [2, 4, 6], 3)
assert result == [[1, 2], [1, 4], [1, 6]], f"Got {result}"
print(f"\n[Test 1] [1,7,11] [2,4,6] k=3 → {result}  (sums: 3,5,7)")
print("✅ Test 1 PASSED")

# Test 2: Duplicates
result = k_smallest_pairs([1, 1, 2], [1, 2, 3], 2)
assert len(result) == 2 and result[0] == [1, 1]
print(f"\n[Test 2] Duplicates → {result}")
print("✅ Test 2 PASSED")

# Test 3: k larger than available pairs
result = k_smallest_pairs([1], [1, 2], 5)
assert len(result) == 2   # only 2 pairs exist
print(f"\n[Test 3] k > available pairs → {result}  (only {len(result)} pairs)")
print("✅ Test 3 PASSED")

# Test 4: Empty arrays
result = k_smallest_pairs([], [1, 2], 3)
assert result == []
print(f"\n[Test 4] Empty array → {result}")
print("✅ Test 4 PASSED")

# Test 5: k=1
result = k_smallest_pairs([1, 2, 3], [1, 2, 3], 1)
assert result == [[1, 1]]
print(f"\n[Test 5] k=1 → {result}  (minimum sum pair)")
print("✅ Test 5 PASSED")

print("\n" + "=" * 60)
print("ALL 5 TESTS PASSED ✅")
print("=" * 60)
