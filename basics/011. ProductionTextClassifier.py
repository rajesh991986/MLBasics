import re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)

class TextClassifier:
    """Complete text classifier with sklearn integration."""
    
    def __init__(self, stopwords=None):
        self.stopwords = stopwords or {'the', 'a', 'is', 'and', 'to'}
        self.vocab = {}
        self.model = MultinomialNB(alpha=1.0)
    
    def preprocess(self, text):
        """Tokenize and clean text."""
        if not isinstance(text, str) or not text:
            return []
        text = re.sub(r'https?://\S+', '', text)
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if t not in self.stopwords]
    
    def build_vocab(self, documents):
        """Build vocabulary from documents."""
        all_tokens = set()
        for doc in documents:
            all_tokens.update(self.preprocess(doc))
        self.vocab = {word: idx for idx, word in enumerate(sorted(all_tokens))}
    
    def vectorize(self, document):
        """Convert document to TF vector."""
        tokens = self.preprocess(document)
        word_counts = Counter(tokens)
        vector = np.zeros(len(self.vocab))
        for word, count in word_counts.items():
            if word in self.vocab:
                vector[self.vocab[word]] = count
        return vector
    
    def fit(self, X_docs, y_labels):
        """Train classifier using sklearn."""
        self.build_vocab(X_docs)
        X_vectors = np.array([self.vectorize(doc) for doc in X_docs])
        self.model.fit(X_vectors, y_labels)  # ← sklearn MultinomialNB
        return self
    
    def predict(self, X_docs):
        """Predict labels using sklearn."""
        X_vectors = np.array([self.vectorize(doc) for doc in X_docs])
        return self.model.predict(X_vectors)  # ← sklearn predict
    
    def evaluate(self, X_test, y_test):
        """Evaluate using sklearn metrics."""
        y_pred = self.predict(X_test)
        
        # Metrics using sklearn
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"\nConfusion Matrix:\n{cm}")
        
        # Detailed report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {'precision': precision, 'recall': recall, 'f1': f1, 'cm': cm}


# ================== USAGE EXAMPLE ==================

if __name__ == '__main__':
    # Sample data
    documents = [
        "this app is great and amazing",
        "terrible app crashes constantly",
        "best purchase ever highly recommend",
        "waste of money doesn't work",
        "love it works perfectly",
        "awful buggy mess"
    ]
    labels = ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
    
    # Split data using sklearn
    X_train, X_test, y_train, y_test = train_test_split(
        documents, labels, 
        test_size=0.33,      # 2 samples for test
        random_state=42,
        stratify=labels      # Maintain pos/neg ratio
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}\n")
    
    # Train classifier
    classifier = TextClassifier()
    classifier.fit(X_train, y_train)
    
    # Evaluate
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    metrics = classifier.evaluate(X_test, y_test)
    
    # Predict new samples
    print("\n" + "=" * 50)
    print("NEW PREDICTIONS")
    print("=" * 50)
    new_docs = ["awesome app", "terrible experience"]
    predictions = classifier.predict(new_docs)
    for doc, pred in zip(new_docs, predictions):
        print(f"'{doc}' → {pred}")
