"""
Two Sum — LC 1

APPROACH: Hashmap — one pass
  For each number, check if complement (target - num) already seen.
  Store value → index as we go.

COMPLEXITY:
  Time:  O(n)  — single pass
  Space: O(n)  — hashmap stores up to n entries

KEY EDGE CASE:
  Same value used twice: [3, 3], target=6 → [0, 1].
  Works because we check complement BEFORE inserting current number.
"""


def two_sum(nums, target):
    """
    Args:
        nums (list): Input array of integers.
        target (int): Target sum.
    Returns:
        list: Indices [i, j] such that nums[i] + nums[j] == target.
    """
    seen = {}   # value → index

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return []   # no solution


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("TWO SUM TESTS")
print("=" * 60)

# Test 1: Standard case
result = two_sum([2, 7, 11, 15], 9)
assert result == [0, 1], f"Expected [0,1], got {result}"
print(f"\n[Test 1] [2,7,11,15] target=9 → {result}")
print("✅ Test 1 PASSED")

# Test 2: Answer not at start
result = two_sum([3, 2, 4], 6)
assert result == [1, 2]
print(f"\n[Test 2] [3,2,4] target=6 → {result}")
print("✅ Test 2 PASSED")

# Test 3: Same value used twice
result = two_sum([3, 3], 6)
assert result == [0, 1], f"Same value edge case, got {result}"
print(f"\n[Test 3] [3,3] target=6 → {result}  (same value, diff index)")
print("✅ Test 3 PASSED")

# Test 4: No solution
result = two_sum([1, 2, 3], 10)
assert result == []
print(f"\n[Test 4] No solution → {result}")
print("✅ Test 4 PASSED")

# Test 5: Negative numbers
result = two_sum([-3, 4, 3, 90], 0)
assert result == [0, 2]
print(f"\n[Test 5] Negative numbers [-3,4,3,90] target=0 → {result}")
print("✅ Test 5 PASSED")

print("\n" + "=" * 60)
print("ALL 5 TESTS PASSED ✅")
print("=" * 60)
