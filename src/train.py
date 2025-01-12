from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import optuna
import os
import joblib

def load_data():
    """Load and split the Digits dataset."""
    data = load_digits()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    """Define the objective function for Optuna hyperparameter tuning."""
    # Suggest hyperparameters for RandomForestClassifier
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
    max_depth = trial.suggest_int("max_depth", 5, 50, step=5)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    
    # Initialize the model with the suggested hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    
    # Train and evaluate the model using cross-validation
    X_train, _, y_train, _ = load_data()
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    
    # Return the mean of the cross-validation scores
    return scores.mean()

def train_and_save_best_model():
    """Train and save both the baseline and the best model."""
    # Train baseline model (using default hyperparameters)
    X_train, X_test, y_train, y_test = load_data()
    baseline_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    baseline_model.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_test, baseline_model.predict(X_test))
    print(f"Baseline Model Accuracy: {baseline_accuracy * 100:.2f}%")


    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)  # Increase number of trials for better optimization
    best_params = study.best_params
    
    # Train best model using Optuna's best hyperparameters
    best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)
    best_accuracy = accuracy_score(y_test, best_model.predict(X_test))

    # Print the accuracies of both models
    print(f"Baseline Model Accuracy: {baseline_accuracy * 100:.2f}%")
    print(f"Best Model Accuracy: {best_accuracy * 100:.2f}%")
    
    # Get the absolute path of parent directory
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Save best model and baseline model
    baseline_model_path = os.path.join(model_dir, "baseline_model.pkl")
    best_model_path = os.path.join(model_dir, "best_model.pkl")
    
    joblib.dump(baseline_model, baseline_model_path)
    joblib.dump(best_model, best_model_path)

    print(f"Baseline model saved successfully at {baseline_model_path}!")
    print(f"Best model saved successfully at {best_model_path}!")

    print("Best Model Parameters:", best_params)

if __name__ == "__main__":
    train_and_save_best_model()
