"""
EDIT DISTANCE (LEVENSHTEIN) FROM SCRATCH - Apple MLE Interview Solution
=======================================================================

PHASE 1: CLARIFICATION QUESTIONS (Ask interviewer)
--------------------------------------------------
1. "Are we computing minimum edit distance (Levenshtein) or other variants?"
2. "Are all operations (insert, delete, substitute) cost 1, or different costs?"
3. "Should I also return the actual edits, or just the distance?"
4. "Is this case-sensitive comparison?"
5. "Any constraints on string length (for space optimization)?"

PHASE 2: APPROACH (Explain before coding)
-----------------------------------------
"Edit distance is the minimum number of operations to transform string A to B.
Operations: Insert, Delete, Substitute (each costs 1)

I'll use dynamic programming:
- dp[i][j] = edit distance between first i chars of A and first j chars of B
- Base cases: dp[0][j] = j (insert j chars), dp[i][0] = i (delete i chars)
- Recurrence:
    If A[i-1] == B[j-1]: dp[i][j] = dp[i-1][j-1]  (no operation needed)
    Else: dp[i][j] = 1 + min(
        dp[i-1][j],    # delete from A
        dp[i][j-1],    # insert into A
        dp[i-1][j-1]   # substitute
    )

Time: O(m*n), Space: O(m*n), can optimize to O(min(m,n))

Does this approach sound good?"
"""

def edit_distance(str_a, str_b):
    """
    Compute minimum edit distance between two strings.

    Args:
        str_a: Source string
        str_b: Target string

    Returns:
        int: Minimum number of operations (insert, delete, substitute)
    """
    # Input validation
    if str_a is None or str_b is None:
        raise ValueError("Strings cannot be None")

    m, n = len(str_a), len(str_b)

    # Edge cases
    if m == 0:
        return n  # Insert all of str_b
    if n == 0:
        return m  # Delete all of str_a

    # Create DP table
    # dp[i][j] = edit distance between str_a[:i] and str_b[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: transforming empty string
    for i in range(m + 1):
        dp[i][0] = i  # Delete i characters from str_a
    for j in range(n + 1):
        dp[0][j] = j  # Insert j characters into str_a

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str_a[i - 1] == str_b[j - 1]:
                # Characters match, no operation needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Take minimum of three operations
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete from str_a
                    dp[i][j - 1],      # Insert into str_a
                    dp[i - 1][j - 1]   # Substitute
                )

    return dp[m][n]


def edit_distance_optimized(str_a, str_b):
    """
    Space-optimized version using O(min(m,n)) space.
    Only keeps two rows of the DP table.
    """
    if str_a is None or str_b is None:
        raise ValueError("Strings cannot be None")

    # Ensure str_a is the longer string for space optimization
    if len(str_a) < len(str_b):
        str_a, str_b = str_b, str_a

    m, n = len(str_a), len(str_b)

    if n == 0:
        return m

    # Only need previous row and current row
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)

    for i in range(1, m + 1):
        curr_row[0] = i

        for j in range(1, n + 1):
            if str_a[i - 1] == str_b[j - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(
                    prev_row[j],      # Delete
                    curr_row[j - 1],  # Insert
                    prev_row[j - 1]   # Substitute
                )

        # Swap rows
        prev_row, curr_row = curr_row, prev_row

    return prev_row[n]


def edit_distance_with_operations(str_a, str_b):
    """
    Returns both the distance and the sequence of operations.
    Useful for explaining the solution.
    """
    if str_a is None or str_b is None:
        raise ValueError("Strings cannot be None")

    m, n = len(str_a), len(str_b)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str_a[i - 1] == str_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack to find operations
    operations = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and str_a[i - 1] == str_b[j - 1]:
            operations.append(f"MATCH '{str_a[i-1]}'")
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            operations.append(f"SUBSTITUTE '{str_a[i-1]}' -> '{str_b[j-1]}'")
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            operations.append(f"INSERT '{str_b[j-1]}'")
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            operations.append(f"DELETE '{str_a[i-1]}'")
            i -= 1

    return dp[m][n], list(reversed(operations))


# =============================================================
# PHASE 4: TEST CASES
# =============================================================

print("=" * 60)
print("TEST 1: Basic example - 'kitten' to 'sitting'")
print("=" * 60)
str_a, str_b = "kitten", "sitting"
distance = edit_distance(str_a, str_b)
print(f"'{str_a}' -> '{str_b}'")
print(f"Edit distance: {distance} (expected: 3)")
print("Operations: k->s (sub), e->i (sub), +g (insert)")

_, ops = edit_distance_with_operations(str_a, str_b)
print("\nDetailed operations:")
for op in ops:
    print(f"  {op}")

print("\n" + "=" * 60)
print("TEST 2: Same strings (distance = 0)")
print("=" * 60)
distance = edit_distance("apple", "apple")
print(f"'apple' -> 'apple': {distance} (expected: 0)")

print("\n" + "=" * 60)
print("TEST 3: EDGE CASE - Empty string to non-empty")
print("=" * 60)
distance = edit_distance("", "abc")
print(f"'' -> 'abc': {distance} (expected: 3 - insert a,b,c)")
distance = edit_distance("abc", "")
print(f"'abc' -> '': {distance} (expected: 3 - delete a,b,c)")

print("\n" + "=" * 60)
print("TEST 4: EDGE CASE - Both empty")
print("=" * 60)
distance = edit_distance("", "")
print(f"'' -> '': {distance} (expected: 0)")

print("\n" + "=" * 60)
print("TEST 5: EDGE CASE - Single character")
print("=" * 60)
print(f"'a' -> 'a': {edit_distance('a', 'a')} (expected: 0)")
print(f"'a' -> 'b': {edit_distance('a', 'b')} (expected: 1 - substitute)")
print(f"'a' -> '': {edit_distance('a', '')} (expected: 1 - delete)")
print(f"'' -> 'a': {edit_distance('', 'a')} (expected: 1 - insert)")

print("\n" + "=" * 60)
print("TEST 6: Completely different strings")
print("=" * 60)
distance = edit_distance("abc", "xyz")
print(f"'abc' -> 'xyz': {distance} (expected: 3 - substitute all)")

print("\n" + "=" * 60)
print("TEST 7: Only insertions needed")
print("=" * 60)
distance = edit_distance("ac", "abc")
print(f"'ac' -> 'abc': {distance} (expected: 1 - insert b)")

print("\n" + "=" * 60)
print("TEST 8: Only deletions needed")
print("=" * 60)
distance = edit_distance("abc", "ac")
print(f"'abc' -> 'ac': {distance} (expected: 1 - delete b)")

print("\n" + "=" * 60)
print("TEST 9: Space-optimized version comparison")
print("=" * 60)
test_pairs = [("intention", "execution"), ("algorithm", "altruistic"), ("saturday", "sunday")]
for a, b in test_pairs:
    d1 = edit_distance(a, b)
    d2 = edit_distance_optimized(a, b)
    print(f"'{a}' -> '{b}': standard={d1}, optimized={d2}, match={d1==d2}")

print("\n" + "=" * 60)
print("TEST 10: NLP use case - Spell checking / fuzzy matching")
print("=" * 60)
query = "aple"
candidates = ["apple", "apply", "maple", "ape", "application"]
print(f"Query: '{query}'")
print("Finding closest matches:")
distances = [(word, edit_distance(query, word)) for word in candidates]
distances.sort(key=lambda x: x[1])
for word, dist in distances:
    print(f"  '{word}': distance={dist}")

print("\n" + "=" * 60)
print("TEST 11: Unicode characters")
print("=" * 60)
distance = edit_distance("café", "cafe")
print(f"'café' -> 'cafe': {distance} (expected: 1)")

print("""
\n=============================================================
PHASE 5: COMPLEXITY ANALYSIS
=============================================================
Standard DP:
  Time: O(m * n) where m, n are string lengths
  Space: O(m * n) for the full DP table

Space-Optimized:
  Time: O(m * n) - same
  Space: O(min(m, n)) - only two rows

With Operations:
  Time: O(m * n) for DP + O(m + n) for backtracking
  Space: O(m * n) + O(m + n) for operations list

NLP APPLICATIONS AT APPLE:
1. Spell correction in search queries
2. Fuzzy matching for content rights (similar titles)
3. Detecting near-duplicate content
4. Autocomplete suggestions ranking
5. OCR error correction

PRODUCTION IMPROVEMENTS:
1. Early termination if distance exceeds threshold
2. Use Damerau-Levenshtein (includes transpositions)
3. Weighted edit costs (typo patterns vary by keyboard layout)
4. Preprocessing: lowercase, remove accents for comparison
5. Caching for repeated queries
6. Parallelization for batch comparisons
7. Use BK-trees for efficient nearest neighbor search
""")
