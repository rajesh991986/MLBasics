"""
CONFUSION MATRIX FROM SCRATCH - Apple MLE Interview Solution
============================================================

PHASE 1: CLARIFICATION QUESTIONS (Ask interviewer)
--------------------------------------------------
1. "Are y_true and y_pred lists of integers representing class labels?"
2. "Should I handle multi-class or just binary classification?"
3. "What should I return - a 2D list/matrix or numpy array?"
4. "Should I also return the labels in sorted order?"

PHASE 2: APPROACH (Explain before coding)
-----------------------------------------
"I'll build a confusion matrix where:
- Rows represent actual classes
- Columns represent predicted classes  
- Cell [i][j] = count of samples with true label i, predicted as j

Time Complexity: O(n + k²)
  - O(n) to iterate through y_true and y_pred
  - O(k) to find unique classes (where k = number of unique classes)
  - O(k²) to initialize matrix
  - Overall: O(n + k²), optimal for this problem

Space Complexity: O(k²)
  - O(k²) for confusion matrix
  - O(k) for class_to_idx mapping
  - Overall: O(k²), optimal
Does this approach sound good before I start coding?"
"""

def confusion_matrix(y_true, y_pred):
    """
    Build confusion matrix from scratch.

    Args:
        y_true: List of true labels
        y_pred: List of predicted labels

    Returns:
        matrix: 2D list where matrix[i][j] = count of true=i, pred=j
        labels: Sorted list of unique labels
    """
    # Input validation
    if y_true is None or y_pred is None:
        raise ValueError("Inputs cannot be None")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    if len(y_true) == 0:
        return [], []

    # Get sorted unique labels from both arrays
    labels = sorted(set(y_true) | set(y_pred))
    n_labels = len(labels)

    # Create label to index mapping for O(1) lookup
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    # Initialize matrix with zeros
    matrix = [[0] * n_labels for _ in range(n_labels)]

    # Fill the matrix - single pass O(n)
    for true, pred in zip(y_true, y_pred):
        true_idx = label_to_idx[true]
        pred_idx = label_to_idx[pred]
        matrix[true_idx][pred_idx] += 1

    return matrix, labels


def print_matrix(matrix, labels=None):
    """Clean print for interview - shows actual/predicted clearly"""
    if not matrix:
        return print("Empty")
    
    k = len(matrix)
    # Just print the 2D list directly - interviewer can read it
    print(f"Confusion Matrix ({k}x{k}):")
    for row in matrix:
        print(f"  {row}")



# =============================================================
# PHASE 4: TEST CASES (Run through these with interviewer)
# =============================================================

print("=" * 60)
print("TEST 1: Basic binary classification")
print("=" * 60)
y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1, 1, 1]
print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
matrix, labels = confusion_matrix(y_true, y_pred)
print_confusion_matrix(matrix, labels)
print(f"\nExpected: TN=2, FP=1, FN=1, TP=4")
print(f"Got: TN={matrix[0][0]}, FP={matrix[0][1]}, FN={matrix[1][0]}, TP={matrix[1][1]}")

print("\n" + "=" * 60)
print("TEST 2: Multi-class (3 classes)")
print("=" * 60)
y_true = [0, 0, 1, 1, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1]
matrix, labels = confusion_matrix(y_true, y_pred)
print_confusion_matrix(matrix, labels)

print("\n" + "=" * 60)
print("TEST 3: EDGE CASE - Empty input")
print("=" * 60)
matrix, labels = confusion_matrix([], [])
print(f"Result: matrix={matrix}, labels={labels}")

print("\n" + "=" * 60)
print("TEST 4: EDGE CASE - Single element")
print("=" * 60)
matrix, labels = confusion_matrix([1], [1])
print_confusion_matrix(matrix, labels)

print("\n" + "=" * 60)
print("TEST 5: EDGE CASE - All wrong predictions")
print("=" * 60)
y_true = [0, 0, 1, 1]
y_pred = [1, 1, 0, 0]
matrix, labels = confusion_matrix(y_true, y_pred)
print_confusion_matrix(matrix, labels)

print("\n" + "=" * 60)
print("TEST 6: EDGE CASE - String labels")
print("=" * 60)
y_true = ["cat", "cat", "dog", "dog"]
y_pred = ["cat", "dog", "dog", "dog"]
matrix, labels = confusion_matrix(y_true, y_pred)
print_confusion_matrix(matrix, labels)

print("\n" + "=" * 60)
print("TEST 7: EDGE CASE - Label in pred but not in true")
print("=" * 60)
y_true = [0, 0, 0]
y_pred = [0, 1, 0]  # Class 1 never appears in true
matrix, labels = confusion_matrix(y_true, y_pred)
print_confusion_matrix(matrix, labels)

print("""
\n=============================================================
PHASE 5: COMPLEXITY ANALYSIS
=============================================================
Time Complexity: O(n + k*log(k))
  - O(n) to iterate through all samples
  - O(k*log(k)) to sort unique labels (k = num classes)
  - For most cases k << n, so effectively O(n)

Space Complexity: O(k^2 + n)
  - O(k^2) for the confusion matrix
  - O(n) for the set of labels (worst case all unique)

PRODUCTION IMPROVEMENTS:
1. Add input type validation (check if lists/arrays)
2. Support numpy arrays for better performance
3. Add normalization option (row-wise, column-wise, all)
4. Add option to return as pandas DataFrame
5. Add logging for debugging
6. Support sparse matrices for many classes
""")
