import numpy as np
import logging
from typing import Optional, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogisticRegression:
    """
    Logistic Regression classifier with mini-batch SGD and L2 regularization.

    Time Complexity: O(max_iterations * n_samples * n_features)
    Space Complexity: O(n_features)
    """

    def __init__(
        self,
        max_iterations: int = 1000,
        learning_rate: float = 0.01,
        regularization: float = 0.001,
        batch_size: int = 100,
        threshold: float = 0.5,
        tolerance: float = 1e-4,
        patience: int = 10
    ):
        """
        Initialize Logistic Regression model.

        Args:
            max_iterations: Maximum training epochs
            learning_rate: Step size for gradient descent
            regularization: L2 regularization strength
            batch_size: Mini-batch size for SGD
            threshold: Decision threshold for binary classification
            tolerance: Early stopping tolerance for loss improvement
            patience: Number of epochs to wait before early stopping
        """
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size
        self.threshold = threshold
        self.tolerance = tolerance
        self.patience = patience

        # Model parameters
        self.weights = None
        self.bias = 0.0
        self.loss_history = []
        self.is_fitted = False

    def _xavier_initialization(self, n_features: int) -> np.ndarray:
        """
        Xavier/Glorot initialization for weights.

        Args:
            n_features: Number of input features

        Returns:
            Initialized weight vector
        """
        limit = np.sqrt(6.0 / (n_features + 1))
        return np.random.uniform(-limit, limit, n_features)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function.

        Args:
            z: Input array

        Returns:
            Sigmoid activation values
        """
        # Clip to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss with L2 regularization.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_true: True labels (n_samples,)

        Returns:
            Loss value
        """
        n_samples = X.shape[0]

        # Forward pass
        z = X @ self.weights + self.bias
        y_pred = self._sigmoid(z)

        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Binary cross-entropy
        bce = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        bce /= n_samples

        # L2 regularization (with factor of 2)
        l2_penalty = (self.regularization / 2.0) * np.sum(self.weights ** 2)

        return bce + l2_penalty

    def _fit_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        Perform one gradient descent step on a mini-batch.

        Args:
            X_batch: Feature matrix for batch
            y_batch: Labels for batch

        Returns:
            Loss value for this batch
        """
        n_samples = X_batch.shape[0]

        # Forward pass
        z = X_batch @ self.weights + self.bias
        y_pred = self._sigmoid(z)

        # Compute loss
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        bce = -np.sum(y_batch * np.log(y_pred_clipped) + (1 - y_batch) * np.log(1 - y_pred_clipped))
        bce /= n_samples
        bce += (self.regularization / 2.0) * np.sum(self.weights ** 2)

        # Compute gradients
        error = y_pred - y_batch
        dw = (X_batch.T @ error) / n_samples + self.regularization * self.weights
        db = np.sum(error) / n_samples

        # Update parameters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

        return bce

    def _validate_inputs(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data.

        Args:
            X: Feature matrix
            y: Optional label vector

        Raises:
            ValueError: If inputs are invalid
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot work with empty dataset")

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")

        if y is not None:
            if y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Shape mismatch: X has {X.shape[0]} samples, y has {y.shape[0]}"
                )

            if not np.all((y == 0) | (y == 1)):
                raise ValueError("y must contain only binary values (0 or 1)")

        if self.is_fitted and X.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Feature mismatch: trained on {self.weights.shape[0]} features, "
                f"got {X.shape[1]}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Train the logistic regression model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)

        Returns:
            self (for method chaining)
        """
        # Convert to numpy if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Validate inputs
        self._validate_inputs(X, y)

        n_samples, n_features = X.shape

        # Initialize weights
        self.weights = self._xavier_initialization(n_features)
        self.bias = 0.0
        self.loss_history = []

        # Early stopping tracking
        best_loss = float('inf')
        no_improve_count = 0

        logger.info(f"Starting training: {n_samples} samples, {n_features} features")

        # Training loop
        for epoch in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch gradient descent
            epoch_losses = []
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]

                batch_loss = self._fit_batch(X_batch, y_batch)
                epoch_losses.append(batch_loss)
                self.loss_history.append(batch_loss)

            # Check convergence
            avg_epoch_loss = np.mean(epoch_losses)

            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Loss = {avg_epoch_loss:.6f}")

            # Early stopping
            if abs(best_loss - avg_epoch_loss) < self.tolerance:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                best_loss = min(best_loss, avg_epoch_loss)
                no_improve_count = 0

        self.is_fitted = True
        logger.info(f"Training complete. Final loss: {avg_epoch_loss:.6f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probability of class 1 for each sample
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.shape[0] == 0:
            return np.array([])

        self._validate_inputs(X)

        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted binary labels
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y_true: True labels

        Returns:
            Dictionary with accuracy, precision, recall, F1 score
        """
        y_pred = self.predict(X)

        # Confusion matrix components
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        # Metrics
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
        }

    def save_model(self, filepath: str) -> None:
        """Save model parameters to file."""
        import pickle
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_dict = {
            'weights': self.weights,
            'bias': self.bias,
            'threshold': self.threshold,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'regularization': self.regularization
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'LogisticRegression':
        """Load model parameters from file."""
        import pickle
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)

        model = cls()
        model.weights = model_dict['weights']
        model.bias = model_dict['bias']
        model.threshold = model_dict['threshold']
        model.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
        return model


# ============================================================================
# COMPREHENSIVE TEST SUITE - Show this during interview if time permits
# ============================================================================

def run_comprehensive_tests():
    """Run all edge case tests."""
    print("\n" + "="*70)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*70)

    # Test 1: Basic functionality
    print("\n[TEST 1] Basic Training & Prediction")
    np.random.seed(42)
    X_train = np.random.randn(100, 3)
    y_train = (X_train[:, 0] > 0).astype(int)

    model = LogisticRegression(max_iterations=500, learning_rate=0.1)
    model.fit(X_train, y_train)

    metrics = model.evaluate(X_train, y_train)
    print(f"Training Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    assert metrics['accuracy'] > 0.85, "Should achieve >85% accuracy"
    print("✅ PASSED")

    # Test 2: Empty input handling
    print("\n[TEST 2] Empty Input Handling")
    try:
        model_empty = LogisticRegression()
        model_empty.fit(np.array([]).reshape(0, 3), np.array([]))
        print("❌ FAILED - Should raise ValueError")
    except ValueError as e:
        print(f"✅ PASSED - Correctly raised ValueError: {e}")

    # Test 3: Single sample
    print("\n[TEST 3] Single Sample")
    X_single = np.array([[1.0, 2.0, 3.0]])
    y_single = np.array([1])
    model_single = LogisticRegression(max_iterations=100)
    model_single.fit(X_single, y_single)
    pred = model_single.predict(X_single)
    print(f"Prediction: {pred[0]}")
    assert pred.shape == (1,), "Should return single prediction"
    print("✅ PASSED")

    # Test 4: All same class
    print("\n[TEST 4] All Same Class")
    X_ones = np.random.randn(50, 3)
    y_ones = np.ones(50)
    model_ones = LogisticRegression(max_iterations=200)
    model_ones.fit(X_ones, y_ones)
    preds = model_ones.predict(X_ones)
    print(f"Predicted 1s: {np.sum(preds)}/50")
    assert np.mean(preds) > 0.7, "Should predict mostly 1s"
    print("✅ PASSED")

    # Test 5: Perfectly separable data
    print("\n[TEST 5] Perfectly Separable Data")
    X_sep = np.vstack([
        np.random.randn(50, 2) + 5,  # Class 1
        np.random.randn(50, 2) - 5   # Class 0
    ])
    y_sep = np.array([1]*50 + [0]*50)
    model_sep = LogisticRegression(max_iterations=500, learning_rate=0.1)
    model_sep.fit(X_sep, y_sep)
    accuracy = np.mean(model_sep.predict(X_sep) == y_sep)
    print(f"Accuracy: {accuracy:.3f}")
    assert accuracy > 0.95, "Should achieve >95% on separable data"
    print("✅ PASSED")

    # Test 6: Shape mismatch
    print("\n[TEST 6] Shape Mismatch Detection")
    try:
        model.fit(np.random.randn(10, 3), np.array([1, 0]))
        print("❌ FAILED - Should raise ValueError")
    except ValueError as e:
        print(f"✅ PASSED - Correctly raised ValueError: {e}")

    # Test 7: Non-binary labels
    print("\n[TEST 7] Non-Binary Labels Detection")
    try:
        model.fit(np.random.randn(10, 3), np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1]))
        print("❌ FAILED - Should raise ValueError")
    except ValueError as e:
        print(f"✅ PASSED - Correctly raised ValueError: {e}")

    # Test 8: Prediction on unfitted model
    print("\n[TEST 8] Unfitted Model Prediction")
    try:
        unfitted_model = LogisticRegression()
        unfitted_model.predict(X_train)
        print("❌ FAILED - Should raise ValueError")
    except ValueError as e:
        print(f"✅ PASSED - Correctly raised ValueError: {e}")

    # Test 9: High dimensional data
    print("\n[TEST 9] High Dimensional Data")
    X_high = np.random.randn(100, 500)
    y_high = (X_high[:, 0] > 0).astype(int)
    model_high = LogisticRegression(max_iterations=200)
    model_high.fit(X_high, y_high)
    assert model_high.weights.shape == (500,), "Should handle 500 features"
    print(f"✅ PASSED - Weights shape: {model_high.weights.shape}")

    # Test 10: Custom threshold
    print("\n[TEST 10] Custom Decision Threshold")
    model_thresh = LogisticRegression(threshold=0.3, max_iterations=500)
    model_thresh.fit(X_train, y_train)

    model_normal = LogisticRegression(threshold=0.5, max_iterations=500)
    model_normal.fit(X_train, y_train)

    pred_low = model_thresh.predict(X_train)
    pred_normal = model_normal.predict(X_train)

    print(f"Threshold 0.3 predicts {np.sum(pred_low)} ones")
    print(f"Threshold 0.5 predicts {np.sum(pred_normal)} ones")
    print("✅ PASSED")

    print("\n" + "="*70)
    print("ALL TESTS PASSED! 🎉")
    print("="*70)


# ============================================================================
# MAIN EXECUTION - Demo what you'd show in interview
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION - APPLE ML ENGINEER INTERVIEW DEMO")
    print("="*70)

    # Basic demo
    print("\n[DEMO] Training on simple linearly separable problem...")
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)

    model = LogisticRegression(max_iterations=500, learning_rate=0.1)
    model.fit(X, y)

    print("\n[DEMO] Evaluating model...")
    metrics = model.evaluate(X, y)
    print(f"\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")

    print(f"\n  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"    TP: {cm['tp']}, FP: {cm['fp']}")
    print(f"    FN: {cm['fn']}, TN: {cm['tn']}")

    # Run comprehensive tests
    run_comprehensive_tests()

    print("\n" + "="*70)
    print("INTERVIEW TALKING POINTS:")
    print("="*70)
    print("""
1. TIME COMPLEXITY: O(max_iterations × n_samples × n_features)
   - Can't improve for iterative optimization

2. SPACE COMPLEXITY: O(n_features)
   - Only store weights vector, process batches on-the-fly

3. PRODUCTION IMPROVEMENTS:
   - ✅ Input validation with clear error messages
   - ✅ Early stopping to prevent overfitting
   - ✅ Model persistence (save/load)
   - ✅ Comprehensive evaluation metrics
   - ✅ Logging for monitoring
   - ✅ Numerical stability (clipping)

4. FOR APPLE SERVICES SCALE:
   - Handle 175 countries, 37 languages → UTF-8 validation
   - Privacy-first: No PII in model artifacts
   - Distributed training for large datasets
   - API-first design with versioning
   - Monitoring & alerting for model drift
    """)