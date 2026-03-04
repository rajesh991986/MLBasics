"""
K-Means Clustering - Interview Practice Template
Fixed version with all improvements, ready for timed practice.
"""

import numpy as np
import time


class KMeans:
    """K-Means clustering with K-means++ init and mini-batch support."""
    
    def __init__(self, k=3, max_iters=100, tol=1e-4, init='k-means++'):
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    
    def interia(self,X):
        distance = np.linalg.norm(X[:,None]-self.centroids[None,:],axis=2)
        self.inertia = np.sum(np.min(distance,axis=1)**2)
        
    def _init_centroids(self, X):
        """Initialize centroids using k-means++ or random."""
        if self.init == 'k-means++':
            centroids = [X[np.random.choice(len(X))]]
            for _ in range(1, self.k):
                dists = np.linalg.norm(X[:, None] - centroids, axis=2)
                min_dists = np.min(dists, axis=1)
                probs = min_dists ** 2
                prob_sum = probs.sum()
                if prob_sum == 0:  # FIX: All points identical
                    probs = np.ones(len(X)) / len(X)
                else:
                    probs /= prob_sum
                centroids.append(X[np.random.choice(len(X), p=probs)])
            return np.array(centroids)
        else:
            idx = np.random.choice(len(X), self.k, replace=False)
            return X[idx]
    
    def fit(self, X):
        """Fit K-means model. Time: O(n*k*d*iters), Space: O(n*k)"""
        # Validation
        if X is None:
            raise ValueError("X cannot be None")
        X = np.asarray(X, dtype=np.float64)
        if X.size == 0:
            raise ValueError("X cannot be empty")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        if np.any(~np.isfinite(X)):  # FIX: Check for NaN/inf
            raise ValueError("X contains NaN or inf values")
        if self.k > len(X):
            raise ValueError(f"k={self.k} cannot exceed n_samples={len(X)}")
        
        # Initialize
        self.centroids = self._init_centroids(X)
        
        for i in range(self.max_iters):
            # Assign clusters
            distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if (labels == j).any() 
                else X[np.argmax(np.min(distances, axis=1))]  # Reassign empty
                for j in range(self.k)
            ])
            
            # Check convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                self.n_iter_ = i + 1
                self.centroids = new_centroids
                break
            
            self.centroids = new_centroids
        else:
            self.n_iter_ = self.max_iters
        
        # Final assignments
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        self.labels_ = np.argmin(distances, axis=1)
        self.inertia_ = np.sum(np.min(distances, axis=1) ** 2)
        
        return self
    
    def fit_minibatch(self, X, batch_size=100):
        """Mini-batch K-means for memory efficiency (on-device use)."""
        if X is None or X.size == 0:
            raise ValueError("X cannot be None or empty")
        X = np.asarray(X, dtype=np.float64)
        if np.any(~np.isfinite(X)):
            raise ValueError("X contains NaN or inf")
        
        n_samples = len(X)
        batch_size = min(batch_size, n_samples)
        self.centroids = self._init_centroids(X[:batch_size])
        
        for i in range(self.max_iters):
            idx = np.random.choice(n_samples, batch_size, replace=False)
            X_batch = X[idx]
            
            distances = np.linalg.norm(X_batch[:, None] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            learning_rate = 1.0 / (i + 1)
            for j in range(self.k):
                cluster_mask = labels == j
                if cluster_mask.any():
                    new_centroid = X_batch[cluster_mask].mean(axis=0)
                    self.centroids[j] = (1 - learning_rate) * self.centroids[j] + learning_rate * new_centroid
        
        # Final full pass
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        self.labels_ = np.argmin(distances, axis=1)
        self.inertia_ = np.sum(np.min(distances, axis=1) ** 2)
        self.n_iter_ = self.max_iters
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        if self.centroids is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X)
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def save(self, filepath):
        """Save model to disk (Apple deployment)."""
        if self.centroids is None:
            raise RuntimeError("Model not fitted")
        np.savez_compressed(
            filepath,
            centroids=self.centroids,
            k=self.k,
            tol=self.tol,
            inertia=self.inertia_
        )
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk."""
        data = np.load(filepath)
        model = cls(k=int(data['k']), tol=float(data['tol']))
        model.centroids = data['centroids']
        model.inertia_ = float(data['inertia']) if 'inertia' in data else None
        return model
    
    def quantize(self):
        """Quantize centroids to INT8 (4x compression for on-device)."""
        if self.centroids is None:
            raise RuntimeError("Model not fitted")
        c_min, c_max = self.centroids.min(), self.centroids.max()
        centroids_scaled = (self.centroids - c_min) / (c_max - c_min + 1e-8)
        centroids_int8 = (centroids_scaled * 127).astype(np.int8)
        return centroids_int8, c_min, c_max


def find_optimal_k(X, k_range=range(2, 11)):
    """Elbow method to find optimal K."""
    inertias = []
    for k in k_range:
        model = KMeans(k=k)
        model.fit(X)
        inertias.append(model.inertia_)
    return list(k_range), inertias


# ============================================================================
# COMPREHENSIVE TEST SUITE - Practice these edge cases!
# ============================================================================

def test_kmeans():
    """Run all edge case tests."""
    
    print("=" * 60)
    print("K-MEANS EDGE CASE TEST SUITE")
    print("=" * 60)
    
    # Test 1: Normal case
    print("\n✅ Test 1: Normal case (3 clusters)")
    X = np.vstack([
        np.random.normal([2, 2], 0.5, (50, 2)),
        np.random.normal([8, 8], 0.5, (50, 2)),
        np.random.normal([2, 8], 0.5, (50, 2))
    ])
    model = KMeans(k=3)
    model.fit(X)
    print(f"   Inertia: {model.inertia_:.2f}, Converged in {model.n_iter_} iters")
    assert model.centroids.shape == (3, 2)
    
    # Test 2: k = n_samples (boundary)
    print("\n✅ Test 2: k = n_samples")
    X_boundary = np.array([[1, 2], [3, 4], [5, 6]])
    model = KMeans(k=3)
    model.fit(X_boundary)
    assert len(np.unique(model.labels_)) == 3
    print(f"   Each point is its own cluster: {model.labels_}")
    
    # Test 3: All identical points (division by zero fix)
    print("\n✅ Test 3: All identical points")
    X_identical = np.ones((10, 2))
    model = KMeans(k=3)
    model.fit(X_identical)
    assert model.centroids.shape == (3, 2)
    print(f"   Handled without crash, centroids all equal")
    
    # Test 4: Single cluster
    print("\n✅ Test 4: Single cluster (k=1)")
    model_1 = KMeans(k=1)
    model_1.fit(X)
    assert np.allclose(model_1.centroids[0], X.mean(axis=0), atol=0.1)
    print(f"   Centroid matches mean: {model_1.centroids[0]}")
    
    # Test 5: Convergence test
    print("\n✅ Test 5: Early convergence")
    X_conv = np.random.rand(100, 2)
    model_conv = KMeans(k=3, max_iters=100, tol=1e-3)
    model_conv.fit(X_conv)
    assert model_conv.n_iter_ < 100
    print(f"   Converged in {model_conv.n_iter_} iters (< 100 max)")
    
    # Test 6: NaN handling
    print("\n✅ Test 6: NaN validation")
    try:
        X_nan = np.array([[1, 2], [np.nan, 4]])
        KMeans(k=2).fit(X_nan)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   Caught NaN: {e}")
    
    # Test 7: Empty input
    print("\n✅ Test 7: Empty array")
    try:
        KMeans(k=2).fit(np.array([]))
        assert False
    except ValueError as e:
        print(f"   Caught empty: {e}")
    
    # Test 8: k > n
    print("\n✅ Test 8: k > n_samples")
    try:
        KMeans(k=10).fit(np.array([[1, 2], [3, 4]]))
        assert False
    except ValueError as e:
        print(f"   Caught k>n: {e}")
    
    # Test 9: Mini-batch variant
    print("\n✅ Test 9: Mini-batch K-means")
    X_large = np.random.rand(1000, 10)
    model_mb = KMeans(k=5)
    start = time.time()
    model_mb.fit_minibatch(X_large, batch_size=100)
    print(f"   1000 samples in {time.time()-start:.3f}s, inertia={model_mb.inertia_:.2f}")
    
    # Test 10: Save/load
    print("\n✅ Test 10: Model serialization")
    model.save('/tmp/kmeans_test.npz')
    model_loaded = KMeans.load('/tmp/kmeans_test.npz')
    assert np.allclose(model.centroids, model_loaded.centroids)
    print(f"   Saved and loaded successfully")
    
    # Test 11: Quantization
    print("\n✅ Test 11: INT8 quantization")
    centroids_int8, c_min, c_max = model.quantize()
    print(f"   FP32 shape: {model.centroids.shape}, INT8 shape: {centroids_int8.shape}")
    print(f"   Memory: {model.centroids.nbytes} bytes → {centroids_int8.nbytes} bytes (4x smaller)")
    
    # Test 12: Predict on new data
    print("\n✅ Test 12: Predict on new data")
    X_new = np.array([[2.5, 2.5], [7.5, 7.5]])
    labels_new = model.predict(X_new)
    print(f"   New point labels: {labels_new}")
    
    # Test 13: Elbow method
    print("\n✅ Test 13: Elbow method for optimal K")
    k_values, inertias = find_optimal_k(X, k_range=range(2, 6))
    print(f"   K={k_values}, Inertias={[f'{i:.1f}' for i in inertias]}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)


# ============================================================================
# INTERVIEW PRACTICE - Time yourself on this!
# ============================================================================

if __name__ == "__main__":
    # Run full test suite
    test_kmeans()
    
    # Quick demo
    print("\n" + "=" * 60)
    print("QUICK DEMO")
    print("=" * 60)
    
    X = np.vstack([
        np.random.normal([2, 2], 0.5, (50, 2)),
        np.random.normal([8, 8], 0.5, (50, 2)),
        np.random.normal([2, 8], 0.5, (50, 2))
    ])
    
    model = KMeans(k=3, init='k-means++')
    model.fit(X)
    
    print(f"\nCentroids:\n{model.centroids}")
    print(f"Inertia: {model.inertia_:.2f}")
    print(f"Converged in: {model.n_iter_} iterations")
    print(f"Labels (first 10): {model.labels_[:10]}")
    
    # APPLE INTERVIEW FOLLOW-UPS - Practice answering these:
    print("\n" + "=" * 60)
    print("PRACTICE THESE INTERVIEW QUESTIONS:")
    print("=" * 60)
    print("""
    Q1: "What's the time complexity?"
    A: O(n*k*d*iters) where n=samples, k=clusters, d=dims, iters~10-50
    
    Q2: "How would you deploy this on iPhone with 4GB RAM for 1M users?"
    A: Use fit_minibatch() with batch_size=1000 (processes 4MB at a time)
       Store only centroids (k*d*4 bytes, e.g., 100*128*4 = 50KB)
       Quantize to INT8 for 4x compression → 12.5KB model
    
    Q3: "What if user data must stay private?"
    A: Cluster on-device, only share aggregate stats (cluster sizes + centroids)
       Add differential privacy noise before sharing
       Never send raw user vectors to server
    
    Q4: "Why does your code crash on identical points?"
    A: Division by zero in k-means++ when min_dists.sum()==0
       Fixed: Add uniform fallback if prob_sum==0
    
    Q5: "K-means vs DBSCAN for content clustering?"
    A: K-means: Faster O(n*k*d), small model, requires k
       DBSCAN: Finds arbitrary shapes, but O(n²) time
       For pricing tiers (fixed k, global scale) → K-means wins
    """)