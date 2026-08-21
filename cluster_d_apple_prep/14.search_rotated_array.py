"""
Search in Rotated Sorted Array — LC 33

APPROACH: Modified binary search
  One half is always sorted. Determine which half, check if target falls there.
  Left half sorted iff nums[left] <= nums[mid].

COMPLEXITY:
  Time:  O(log n)
  Space: O(1)

TRICKY CASE: [3, 1] target=1
  mid=0, nums[mid]=3, nums[left]=3 → left half sorted (3 <= 3).
  target=1 NOT in [3, 3) → go right → left=1, nums[1]=1 == target. ✓

KEY: Use <= not < for left half check to handle single-element windows.
"""


def search(nums, target):
    """
    Args:
        nums (list): Rotated sorted array (no duplicates).
        target (int): Value to find.
    Returns:
        int: Index of target, or -1 if not found.
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        if nums[left] <= nums[mid]:        # left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:                               # right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("SEARCH IN ROTATED SORTED ARRAY TESTS")
print("=" * 60)

# Test 1: Target in left portion
result = search([4, 5, 6, 7, 0, 1, 2], 5)
assert result == 1
print(f"\n[Test 1] [4,5,6,7,0,1,2] target=5 → index {result}")
print("✅ Test 1 PASSED")

# Test 2: Target in right portion
result = search([4, 5, 6, 7, 0, 1, 2], 0)
assert result == 4
print(f"\n[Test 2] target=0 → index {result}")
print("✅ Test 2 PASSED")

# Test 3: Not found
result = search([4, 5, 6, 7, 0, 1, 2], 3)
assert result == -1
print(f"\n[Test 3] target=3 (not found) → {result}")
print("✅ Test 3 PASSED")

# Test 4: Single element found
result = search([1], 1)
assert result == 0
print(f"\n[Test 4] [1] target=1 → {result}")
print("✅ Test 4 PASSED")

# Test 5: Single element not found
result = search([1], 0)
assert result == -1
print(f"\n[Test 5] [1] target=0 → {result}")
print("✅ Test 5 PASSED")

# Test 6: Tricky case [3, 1]
result = search([3, 1], 1)
assert result == 1, f"Expected 1, got {result}"
print(f"\n[Test 6] [3,1] target=1 → index {result}  (tricky: left half is [3,3), target not in it)")
print("✅ Test 6 PASSED")

# Test 7: Not rotated (already sorted)
result = search([1, 2, 3, 4, 5], 3)
assert result == 2
print(f"\n[Test 7] Not rotated [1,2,3,4,5] target=3 → index {result}")
print("✅ Test 7 PASSED")

print("\n" + "=" * 60)
print("ALL 7 TESTS PASSED ✅")
print("=" * 60)
