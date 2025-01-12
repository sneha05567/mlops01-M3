import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, accuracy_score

def load_data():
    """Load the Digits dataset."""
    data = load_digits()
    X, y = data.data, data.target
    return X, y

def evaluate_model():
    """Evaluate the best model on the Digits dataset."""
    # Load the best model using joblib
    model = joblib.load("models/best_model.pkl")

    X, y = load_data()
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=[str(i) for i in range(10)])

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

if __name__ == "__main__":
    evaluate_model()
