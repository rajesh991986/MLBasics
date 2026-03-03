import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ============================================================
# STEP 1: MODEL
# nn.Module requires __init__ and forward()
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()   # NEVER forget this — breaks backprop silently

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            # NO softmax here — CrossEntropyLoss applies softmax internally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================
# STEP 2: TRAINING LOOP
# The 5-line pattern you must know cold:
#   1. optimizer.zero_grad()
#   2. outputs = model(X)
#   3. loss = criterion(outputs, y)
#   4. loss.backward()
#   5. optimizer.step()
# ============================================================

def train(model: nn.Module,
          X: torch.Tensor,
          y: torch.Tensor,
          lr: float = 0.01,
          epochs: int = 10) -> None:

    criterion = nn.CrossEntropyLoss()       # softmax + NLL, expects raw logits
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train()   # sets dropout/batchnorm to training mode (good habit)

    for epoch in range(epochs):
        optimizer.zero_grad()              # 1. clear gradients from last step

        outputs = model(X)                 # 2. forward pass → raw logits

        loss = criterion(outputs, y)       # 3. compute loss
                                           #    CrossEntropyLoss = softmax + NLL
                                           #    expects (batch, classes) vs (batch,)

        loss.backward()                    # 4. backprop — compute all gradients

        optimizer.step()                   # 5. update weights

        if (epoch + 1) % 20 == 0:
            preds = outputs.argmax(dim=-1)
            accuracy = (preds == y).float().mean().item()
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Acc: {accuracy:.3f}")


# ============================================================
# STEP 3: EVALUATION (separate from training — no gradients)
# ============================================================

def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()   # disables dropout, uses running stats for batchnorm

    with torch.no_grad():   # no gradient tracking — saves memory, faster
        outputs = model(X)
        preds = outputs.argmax(dim=-1)
        accuracy = (preds == y).float().mean().item()

    return accuracy


def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Return predicted class labels."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
    return logits.argmax(dim=-1)


# ============================================================
# STEP 4: TOY DATA + PUTTING IT ALL TOGETHER
# ============================================================

def make_toy_data(n_samples=200, input_dim=10, n_classes=3, seed=42):
    """Simple linearly separable toy data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    # Make classes separable: shift each class mean
    y = np.array([i % n_classes for i in range(n_samples)])
    for c in range(n_classes):
        X[y == c] += c * 2.0
    return X, y


if __name__ == "__main__":
    # Config
    INPUT_DIM  = 10
    HIDDEN_DIM = 32
    N_CLASSES  = 3
    EPOCHS     = 100
    LR         = 0.01

    # Data
    X_np, y_np = make_toy_data(n_samples=200, input_dim=INPUT_DIM, n_classes=N_CLASSES)

    # Train/val split (manual — no sklearn)
    split = int(0.8 * len(X_np))
    X_train = torch.tensor(X_np[:split], dtype=torch.float32)
    X_val   = torch.tensor(X_np[split:], dtype=torch.float32)
    y_train = torch.tensor(y_np[:split], dtype=torch.long)
    y_val   = torch.tensor(y_np[split:], dtype=torch.long)

    # Model
    model = MLP(INPUT_DIM, HIDDEN_DIM, N_CLASSES)

    # Train
    print("Training...")
    train(model, X_train, y_train, lr=LR, epochs=EPOCHS)

    # Evaluate
    val_acc = evaluate(model, X_val, y_val)
    print(f"\nValidation Accuracy: {val_acc:.3f}")

    # Predict
    predictions = predict(model, X_val)
    print(f"\nPredicted: {predictions.numpy()}")
    print(f"Actual:    {y_val.numpy()}")
