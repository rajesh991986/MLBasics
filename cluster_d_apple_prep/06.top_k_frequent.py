"""
Top K Frequent Elements — LC 347

APPROACH: Counter + min-heap of size K
  Min-heap keeps only the K most frequent. When heap > K, pop the minimum.
  After processing all elements, heap[0] = Kth most frequent.

COMPLEXITY:
  Time:  O(n log k)  — n inserts into heap of size k
  Space: O(n)        — Counter stores all unique elements

APPLE USE CASE:
  "Return the top-K most queried Siri categories this week."
  "Find the K most common error codes in our logs."
"""

from collections import Counter
import heapq


def top_k_frequent(nums, k):
    """
    Args:
        nums (list): Input array.
        k (int): Number of most frequent elements to return.
    Returns:
        list: K most frequent elements (any order).
    """
    count = Counter(nums)
    heap  = []

    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)   # remove least frequent

    return [num for freq, num in heap]


# One-liner alternative (simpler, slightly less efficient):
def top_k_frequent_v2(nums, k):
    return [num for num, _ in Counter(nums).most_common(k)]


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("TOP K FREQUENT ELEMENTS TESTS")
print("=" * 60)

# Test 1: Standard case
result = set(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
assert result == {1, 2}, f"Expected {{1,2}}, got {result}"
print(f"\n[Test 1] [1,1,1,2,2,3] k=2 → {result}")
print("✅ Test 1 PASSED")

# Test 2: Single element
result = top_k_frequent([1], 1)
assert result == [1]
print(f"\n[Test 2] [1] k=1 → {result}")
print("✅ Test 2 PASSED")

# Test 3: k equals all unique elements
result = set(top_k_frequent([1, 2], 2))
assert result == {1, 2}
print(f"\n[Test 3] [1,2] k=2 → {result}")
print("✅ Test 3 PASSED")

# Test 4: All same element
result = top_k_frequent([3, 3, 3], 1)
assert result == [3]
print(f"\n[Test 4] [3,3,3] k=1 → {result}")
print("✅ Test 4 PASSED")

# Test 5: Both approaches agree
nums = [4, 1, 2, 2, 3, 3, 3, 4, 4, 4]
r1 = set(top_k_frequent(nums, 2))
r2 = set(top_k_frequent_v2(nums, 2))
assert r1 == r2 == {3, 4}
print(f"\n[Test 5] Both approaches → {r1}")
print("✅ Test 5 PASSED")

print("\n" + "=" * 60)
print("ALL 5 TESTS PASSED ✅")
print("=" * 60)
