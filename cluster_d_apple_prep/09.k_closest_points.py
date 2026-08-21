"""
K Closest Points to Origin — LC 973

APPROACH: Min-heap with squared distance as key
  Skip sqrt — monotonic, doesn't change ranking.
  heapq.nsmallest is the cleanest O(n log k) solution.

COMPLEXITY:
  Time:  O(n log k)  — heap stays size k
  Space: O(k)

APPLE USE CASE:
  "Find K nearest embedding vectors to a query embedding."
  (In practice you'd use FAISS, but the logic is the same.)
"""

import heapq


def k_closest(points, k):
    """
    Args:
        points (list of [x, y]): 2D points.
        k (int): Number of closest points to return.
    Returns:
        list: K closest points to origin [0, 0].
    """
    return heapq.nsmallest(k, points, key=lambda p: p[0]**2 + p[1]**2)


# Manual heap version (to show mechanics if asked):
def k_closest_manual(points, k):
    heap = []
    for x, y in points:
        dist = x*x + y*y
        heapq.heappush(heap, (-dist, [x, y]))   # max-heap via negation
        if len(heap) > k:
            heapq.heappop(heap)
    return [p for _, p in heap]


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("K CLOSEST POINTS TESTS")
print("=" * 60)

# Test 1: Basic case
result = k_closest([[1, 3], [-2, 2]], 1)
assert result == [[-2, 2]], f"Expected [[-2,2]], got {result}"  # dist 8 < 10
print(f"\n[Test 1] [[1,3],[-2,2]] k=1 → {result}  (dist 8 < 10)")
print("✅ Test 1 PASSED")

# Test 2: k=2
result = sorted(k_closest([[3, 3], [5, -1], [-2, 4]], 2))
assert result == sorted([[3, 3], [-2, 4]])
print(f"\n[Test 2] k=2 → {result}")
print("✅ Test 2 PASSED")

# Test 3: Origin point always closest
result = k_closest([[0, 0], [1, 1], [2, 2]], 1)
assert result == [[0, 0]]
print(f"\n[Test 3] Origin [0,0] → {result}")
print("✅ Test 3 PASSED")

# Test 4: k == len(points)
points = [[1, 1], [2, 2], [3, 3]]
result = k_closest(points, 3)
assert len(result) == 3
print(f"\n[Test 4] k = n → all points returned")
print("✅ Test 4 PASSED")

# Test 5: Manual and nsmallest agree
pts = [[3, 3], [5, -1], [-2, 4], [1, 0]]
r1 = sorted(k_closest(pts, 2))
r2 = sorted(k_closest_manual(pts, 2))
assert r1 == r2
print(f"\n[Test 5] Both approaches agree → {r1}")
print("✅ Test 5 PASSED")

print("\n" + "=" * 60)
print("ALL 5 TESTS PASSED ✅")
print("=" * 60)
