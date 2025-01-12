from src.train import load_data, objective, train_and_save_best_model
import unittest
import os

class TestModelTraining(unittest.TestCase):

    def test_load_data(self):
        """Test that data is loaded and split correctly"""
        X_train, X_test, y_train, y_test = load_data()

        # Check that the data is split correctly
        self.assertGreater(X_train.shape[0], 0, "Training data is empty.")
        self.assertGreater(X_test.shape[0], 0, "Test data is empty.")
        self.assertGreater(y_train.shape[0], 0, "Training labels are empty.")
        self.assertGreater(y_test.shape[0], 0, "Test labels are empty.")

    def test_train_and_save_best_model(self):
        """Test that the model is trained and saved successfully"""
        # Assert that the model file has been saved
        self.assertTrue(os.path.exists("models/best_model.pkl"), "Model file not saved successfully.")

if __name__ == "__main__":
    unittest.main()
