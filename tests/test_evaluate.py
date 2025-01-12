import unittest
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from io import StringIO
import sys
from src.evaluate import evaluate_model


class TestModelEvaluation(unittest.TestCase):

    def test_evaluate_model(self):
        """Test that the model achieves an accuracy greater than 95%"""
        # Load the trained model
        with open("models/best_model.pkl", "rb") as f:
            model = joblib.load(f)

        # Load the dataset
        data = load_breast_cancer()
        X, y = data.data, data.target

        # Get predictions
        y_pred = model.predict(X)

        # Evaluate accuracy
        accuracy = accuracy_score(y, y_pred)

        # Assert accuracy is greater than 95%
        self.assertGreater(accuracy, 0.95, f"Accuracy should be greater than 0.95, got {accuracy:.4f}")

    def test_classification_report(self):
        """Test that the classification report is printed correctly"""
        # Capture the output of evaluate_model
        captured_output = StringIO()
        sys.stdout = captured_output

        # Call the evaluate_model function
        evaluate_model()

        # Check if 'classification report' is in the output
        self.assertIn("classification report", captured_output.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
import unittest
import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from io import StringIO
import sys
from src.evaluate import evaluate_model


class TestModelEvaluation(unittest.TestCase):

    def test_evaluate_model(self):
        """Test that the model achieves an accuracy greater than 95%"""
        # Load the trained model
        model = joblib.load("models/best_model.pkl")

        # Load the dataset
        data = load_digits()
        X, y = data.data, data.target

        # Get predictions
        y_pred = model.predict(X)

        # Evaluate accuracy
        accuracy = accuracy_score(y, y_pred)

        # Assert accuracy is greater than 95%
        self.assertGreater(accuracy, 0.95, f"Accuracy should be greater than 0.95, got {accuracy:.4f}")

    def test_classification_report(self):
        """Test that the classification report is printed correctly"""
        # Capture the output of evaluate_model
        captured_output = StringIO()
        sys.stdout = captured_output

        # Call the evaluate_model function
        evaluate_model()

        # Check if 'classification report' is in the output
        self.assertIn("classification report", captured_output.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
