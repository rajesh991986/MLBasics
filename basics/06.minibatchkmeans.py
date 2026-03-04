import numpy as np 

class KMeans:
    def __init__(self, k=3, max_iteration=100, threshold=1e-4, batch_size=20):
        self.k = k
        self.max_iteration = max_iteration
        self.threshold = threshold
        self.batch_size = batch_size
        self.centroids = None  # Changed: None instead of []
        # Remove these for streaming:
        # self.full_data = None
        # self.full_data_size = None
    
    def _inertia(self, X):
        """Calculate inertia for a given dataset."""
        if self.centroids is None:
            raise ValueError("Model not fitted yet")
        distance = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
        min_distance = np.min(distance, axis=1)
        return np.sum(min_distance ** 2)
    
    def _kmeans_plus_plus_batch(self, X):
        """Initialize centroids using K-means++ from a single batch."""
        n = len(X)
        centroids = [X[np.random.randint(n)]]
        
        for _ in range(self.k - 1):
            # Compute distances to existing centroids
            distances = np.linalg.norm(
                X[:, None] - np.array(centroids)[None, :], 
                axis=2
            )
            min_distances = np.min(distances, axis=1)
            
            # Sample proportional to squared distance
            prob = min_distances ** 2
            prob_sum = prob.sum()
            if prob_sum == 0:
                prob = np.ones(n) / n
            else:
                prob /= prob_sum
            
            idx = np.random.choice(n, p=prob)
            centroids.append(X[idx])
        
        return np.array(centroids)
    
    def _fit_mini_batch(self, X_batch):
        """Update centroids using one mini-batch."""
        # Compute distances: (batch_size, k)
        distances = np.linalg.norm(
            X_batch[:, None] - self.centroids[None, :],
            axis=2
        )
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = []
        learning_rate = 0.1
        
        for j in range(self.k):
            mask = (labels == j)
            
            if mask.any():
                # Incremental update
                batch_mean = X_batch[mask].mean(axis=0)
                new_centroid = (
                    (1 - learning_rate) * self.centroids[j] + 
                    learning_rate * batch_mean
                )
                new_centroids.append(new_centroid)
            else:
                # No points in this cluster - keep old centroid
                # OR reassign to random point in batch
                new_centroids.append(self.centroids[j])
        
        return np.array(new_centroids)
    
    def fit(self, X):
        """
        Fit K-means on full dataset (for small datasets that fit in memory).
        
        Time Complexity: O(n_samples * k * n_features * max_iteration)
        Space Complexity: O(n_samples * n_features + k * n_features)
        """
        if X is None or len(X) == 0:
            return self
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if self.k > len(X) or self.k < 1:
            raise ValueError(f"k must be between 1 and {len(X)}, got {self.k}")
        
        n_samples = len(X)
        
        # Initialize centroids
        self.centroids = self._kmeans_plus_plus_batch(X)
        
        # Mini-batch iterations
        for iteration in range(self.max_iteration):
            old_centroids = self.centroids.copy()
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            # Process in mini-batches
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                batch = X_shuffled[i:end_idx]
                self.centroids = self._fit_mini_batch(batch)
            
            # Check convergence
            if np.all(np.abs(old_centroids - self.centroids) < self.threshold):
                print(f"Converged at iteration {iteration}")
                break
        
        return self
    
    # ========== NEW: STREAMING FIT ==========
    def fit_stream(self, data_iterator, init_batch_size=1000):
        """
        Fit K-means on streaming data (for large datasets).
        
        Args:
            data_iterator: Iterator that yields batches of data
            init_batch_size: Number of samples to use for initialization
        
        Time Complexity: O(n_samples * k * n_features * max_iteration)
        Space Complexity: O(k * n_features + batch_size * n_features) ✅
        
        Example:
            def data_generator():
                for i in range(0, 1_000_000, 1000):
                    yield load_data_chunk(i, i+1000)
            
            kmeans.fit_stream(data_generator())
        """
        # Collect initial batch for K-means++ initialization
        init_data = []
        batch_count = 0
        
        try:
            for batch in data_iterator:
                if len(init_data) < init_batch_size:
                    init_data.append(batch)
                    batch_count += 1
                else:
                    break
        except StopIteration:
            pass
        
        if not init_data:
            raise ValueError("data_iterator is empty")
        
        init_data = np.vstack(init_data)
        
        # Initialize centroids from first batches
        self.centroids = self._kmeans_plus_plus_batch(init_data)
        print(f"Initialized centroids from {len(init_data)} samples")
        
        # Now stream through all data
        for iteration in range(self.max_iteration):
            old_centroids = self.centroids.copy()
            batch_count = 0
            
            # Process streaming batches
            for batch in data_iterator:
                if batch is None or len(batch) == 0:
                    continue
                
                if not isinstance(batch, np.ndarray):
                    batch = np.array(batch)
                
                # Update centroids with this batch
                self.centroids = self._fit_mini_batch(batch)
                batch_count += 1
            
            # Check convergence
            centroid_shift = np.linalg.norm(old_centroids - self.centroids)
            if centroid_shift < self.threshold:
                print(f"Converged at iteration {iteration}, processed {batch_count} batches")
                break
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels.
        
        Time Complexity: O(n_samples * k * n_features)
        Space Complexity: O(n_samples * k)
        """
        if self.centroids is None:
            raise ValueError("Model not fitted yet. Call fit() or fit_stream() first.")
        
        if X is None or len(X) == 0:
            return np.array([])
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        distances = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
        return np.argmin(distances, axis=1)


# ========== USAGE EXAMPLES ==========

# Example 1: Your existing code still works (backward compatible)
print("=" * 50)
print("Example 1: Small dataset (existing usage)")
print("=" * 50)

data = np.vstack([
    np.random.normal(loc=[10, 2], scale=0.6, size=(20, 2)),
    np.random.normal(loc=[30, 5], scale=0.6, size=(20, 2)),
    np.random.normal(loc=[50, 10], scale=0.6, size=(20, 2)),
])

kmeans = KMeans(k=3, batch_size=20)
kmeans.fit(data)
print(f"Centroids:\n{kmeans.centroids}")

test_data = np.vstack([
    np.random.normal(loc=[5, 2], scale=0.6, size=(1, 2)),
    np.random.normal(loc=[33, 5], scale=0.6, size=(1, 2)),
    np.random.normal(loc=[48, 10], scale=0.6, size=(1, 2)),
])
print(f"Predictions: {kmeans.predict(test_data)}")

# Example 2: Large dataset with streaming (memory efficient)
print("\n" + "=" * 50)
print("Example 2: Large dataset streaming (1M samples)")
print("=" * 50)

def generate_large_dataset(n_samples=1_000_000, batch_size=1000):
    """Simulate streaming from database/file."""
    for i in range(0, n_samples, batch_size):
        # In reality, you'd load from disk/database here
        batch = np.vstack([
            np.random.normal(loc=[10, 2], scale=2, size=(batch_size // 3, 2)),
            np.random.normal(loc=[50, 5], scale=2, size=(batch_size // 3, 2)),
            np.random.normal(loc=[100, 10], scale=2, size=(batch_size // 3, 2)),
        ])
        yield batch

kmeans_stream = KMeans(k=3, batch_size=1000)
kmeans_stream.fit_stream(generate_large_dataset())
print(f"Centroids:\n{kmeans_stream.centroids}")
print(f"Inertia: {kmeans_stream._inertia(data):.2f}")