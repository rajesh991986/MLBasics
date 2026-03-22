import numpy as np

def relu(z): return np.maximum(0, z)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

class MLP:
    def __init__(self, d_in, d_hid, d_out):
        self.W1 = np.random.randn(d_in,  d_hid) * np.sqrt(2 / d_in)
        self.b1 = np.zeros(d_hid)
        self.W2 = np.random.randn(d_hid, d_out) * np.sqrt(2 / d_hid)
        self.b2 = np.zeros(d_out)

    def forward(self, X):
        self.X  = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.a2 = softmax(self.a1 @ self.W2 + self.b2)
        return self.a2

    def backward(self, y, lr=0.1):
        B = self.X.shape[0]
        dz2 = (self.a2 - y) / B          # CE + softmax gradient simplification
        dW2 = self.a1.T @ dz2
        dz1 = (dz2 @ self.W2.T) * (self.z1 > 0)   # ReLU gate
        dW1 = self.X.T @ dz1

        self.W2 -= lr * dW2;  self.b2 -= lr * dz2.sum(0)
        self.W1 -= lr * dW1;  self.b1 -= lr * dz1.sum(0)

    def fit(self, X, y, epochs=1000, lr=0.5):
        for _ in range(epochs):
            self.forward(X)
            self.backward(y, lr)

    def predict(self, X): return np.argmax(self.forward(X), axis=1)

# XOR smoke test
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[1,0],[0,1],[0,1],[1,0]], dtype=float)
np.random.seed(42)
m = MLP(2, 8, 2)
m.fit(X, y)
assert np.array_equal(m.predict(X), [0,1,1,0]), "XOR failed"
print("XOR passed")
"""
---

## Key things you must still say out loud

Even with the shorter code, Punit will ask about every line. Be ready to narrate:

| Line | What to say |
|------|-------------|
| `z - z.max(...)` | "Subtract row-max for numerical stability — prevents exp overflow" |
| `* np.sqrt(2 / d_in)` | "He initialization — scales variance correctly for ReLU, avoids vanishing/exploding gradients" |
| `(self.a2 - y) / B` | "This is the beautiful simplification — CE loss + softmax Jacobian collapse into just prediction minus label" |
| `* (self.z1 > 0)` | "ReLU gate — gradient is 1 where pre-activation was positive, 0 otherwise" |
| `self.X.T @ dz1` | "Outer product accumulates gradients over the batch" |

---

## Interview writing order (~20 min target)
```
1. softmax + relu     (2 min) — write these first, get them out of the way
2. __init__           (2 min) — He init, zeros for bias
3. forward            (4 min) — cache X, z1, a1, a2 as self attributes
4. backward           (6 min) — narrate dz2 simplification, ReLU gate, chain rule
5. fit / predict      (2 min) — trivial loops
6. XOR test           (4 min) — paste/write, run, assert
"""
