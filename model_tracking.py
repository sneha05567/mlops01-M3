# !pip install mlflow

import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define different hyperparameter configurations
hyperparameter_configs = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 150, "max_depth": None},
]

# Loop through configurations and log each run
for config in hyperparameter_configs:
    n_estimators = config["n_estimators"]
    max_depth = config["max_depth"]

    # Train model with current configuration
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro")
    recall = recall_score(y_test, predictions, average="macro")
    f1 = f1_score(y_test, predictions, average="macro")

    # Prepare data for artifact logging
    predictions_df = pd.DataFrame({
        "True Labels": y_test,
        "Predictions": predictions
    })

    # Infer model signature
    input_example = X_test.head(1)
    signature = infer_signature(X_test, predictions)

    # Start an MLflow run
    with mlflow.start_run():
        # Log model with signature and input example
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        
        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log predictions as an artifact
        predictions_file = f"predictions_{n_estimators}_{max_depth}.csv"
        predictions_df.to_csv(predictions_file, index=False)
        mlflow.log_artifact(predictions_file)
        
        # Log confusion matrix as an artifact
        conf_matrix = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        confusion_matrix_file = f"confusion_matrix_{n_estimators}_{max_depth}.png"
        plt.savefig(confusion_matrix_file)
        mlflow.log_artifact(confusion_matrix_file)
        plt.close()

    print(f"Run complete for n_estimators={n_estimators}, max_depth={max_depth}")

# run the below command
# mlflow ui
# this should give public link for the tracking of the model, metrics, inputs, outputs
