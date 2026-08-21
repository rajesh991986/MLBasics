"""
Kth Largest Element — LC 215

APPROACH: Min-heap of size K
  Maintain a min-heap of size K. After scanning all elements,
  heap[0] = the Kth largest (smallest element in the K largest set).

COMPLEXITY:
  Time:  O(n log k)  — n elements, each insert/replace is O(log k)
  Space: O(k)

ALTERNATIVES:
  Sort: O(n log n) time, O(n) space — simpler to write, fine for small n.
  Quickselect: O(n) average, O(n²) worst — mention if asked for optimal.
"""

import heapq


def find_kth_largest(nums, k):
    """
    Args:
        nums (list): Input array.
        k (int): 1-indexed position (k=1 = largest, k=n = smallest).
    Returns:
        int: Kth largest element.
    """
    heap = nums[:k]
    heapq.heapify(heap)

    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)   # atomic pop+push, O(log k)

    return heap[0]


def find_kth_largest_sort(nums, k):
    """Simple O(n log n) version — mention as alternative."""
    return sorted(nums, reverse=True)[k - 1]


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("KTH LARGEST ELEMENT TESTS")
print("=" * 60)

# Test 1: k=2 standard
result = find_kth_largest([3, 2, 1, 5, 6, 4], 2)
assert result == 5
print(f"\n[Test 1] [3,2,1,5,6,4] k=2 → {result}")
print("✅ Test 1 PASSED")

# Test 2: Larger array
result = find_kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4)
assert result == 4
print(f"\n[Test 2] [3,2,3,1,2,4,5,5,6] k=4 → {result}")
print("✅ Test 2 PASSED")

# Test 3: k=1 → maximum
result = find_kth_largest([2, 1], 1)
assert result == 2
print(f"\n[Test 3] k=1 (maximum) → {result}")
print("✅ Test 3 PASSED")

# Test 4: k=n → minimum
result = find_kth_largest([2, 1], 2)
assert result == 1
print(f"\n[Test 4] k=n (minimum) → {result}")
print("✅ Test 4 PASSED")

# Test 5: Duplicates
result = find_kth_largest([1, 1, 1, 1], 2)
assert result == 1
print(f"\n[Test 5] All duplicates [1,1,1,1] k=2 → {result}")
print("✅ Test 5 PASSED")

# Test 6: Both approaches agree
nums = [3, 2, 1, 5, 6, 4]
assert find_kth_largest(nums, 2) == find_kth_largest_sort(nums, 2)
print(f"\n[Test 6] Heap and sort agree on k=2 → {find_kth_largest(nums, 2)}")
print("✅ Test 6 PASSED")

print("\n" + "=" * 60)
print("ALL 6 TESTS PASSED ✅")
print("=" * 60)
