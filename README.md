## M3: Model Experimentation and Packaging

### **Objective**:
Train a machine learning model, perform hyperparameter tuning, and package the model for deployment.

### Tasks:
**1. Hyperparameter Tuning:**
- Use a library like Optuna or Scikit-learn’s GridSearchCV to perform hyperparameter tuning on a chosen model.
- Document the tuning process and the best parameters found.

**2. Model Packaging:**
- Package the best-performing model using tools like Docker and Flask.
- Create a Dockerfile and a simple Flask application to serve the model.

**Deliverables:**
- A report on hyperparameter tuning results.
- A Dockerfile and Flask application code.
- Screenshots of the model running in a Docker container.

---

This project implements a **handwritten digit recognition** system using a **Random Forest Classifier**, hyperparameter optimized with **Optuna**, deployed via a **Flask API**, and containerized in **Docker** for scalable predictions.

### Project Structure

- **`src/`**: Contains scripts for model training and evaluation.  
  - `train.py`: Trains and saves the best model.  
  - `evaluate.py`: Outputs evaluation metrics.
- **`models/`**: Stores the tuned model (`best_model.pkl`) and baseline model (`baseline_model.pkl`).  
- **`app/`**: Implements the Flask API.  
  - `app.py`: Exposes the `/predict_digit` endpoint for classification. 
- **`tests/`**: Includes unit tests for codebase.  
- **`test-results/`**: Contains testing outputs.  
  - `results.xml`: Pytest results in XML format.
- **`handwritten-digits/`**: Sample 8x8 pixel images (`digit_1.png`, `digit_7.png`, etc.) created using a paint app for testing the API.
- **`.github/workflows/`**: A GitHub Actions pipeline is defined in `main.yml` to automatically build, test, and deploy the application whenever changes are pushed to the `dev` branch or a pull request is made to the `main` branch. 
    - To configure deployment, set the following secrets in your GitHub repository:
        - `DOCKER_USERNAME`: Your Docker Hub username.
        - `DOCKER_PASSWORD`: Your Docker Hub password.
- **`requirements.txt`**: Contains the necessary Python dependencies for the project. Run
```
sudo apt-get update
sudo apt-get install -y libgl1
pip install -r requirements.txt
```
- **`Dockerfile`**: Specifies the setup for the Docker container. It installs dependencies, copies project files, exposes port `5000`, and runs the Flask application serving the model.

---
### Dataset
The dataset used in this project is the **Digits dataset** from `scikit-learn`. It contains 1797 8x8 pixel images of handwritten digits (0-9) used for classification tasks.

### Dataset Description:
- **Number of Instances**: 1797
- **Features**: 64 pixel values (8x8 images)
- **Target**: Multi-class classification (0-9)

The dataset is available directly through `scikit-learn`'s `load_digits()` method.

---

### Model Hyperparameter Tuning
A **Random Forest Classifier** is used for classification in this project. The model is tuned using **Optuna**, a hyperparameter optimization library.

### Hyperparameters Tuned:
- **`n_estimators`**: Number of trees in the forest (range: 50 to 300, step size: 50)
- **`max_depth`**: Maximum depth of the tree (range: 5 to 30, step size: 5)
- **`min_samples_split`**: Minimum number of samples required to split an internal node (range: 2 to 20)
- **`min_samples_leaf`**: Minimum number of samples required to be at a leaf node (range: 1 to 20)

The **Optuna** library is used to find the best hyperparameters for the Random Forest model using cross-validation. 

---

### Model Packaging and Deployment

Once the model is trained and hyperparameters are tuned, it is serialized and packaged for deployment. The deployment process consists of the following steps:

1. **Model Serialization**:
   - The trained Random Forest model is saved using `pickle` to a file (`models/best_model.pkl`). This file is later loaded to make predictions through the Flask API.

2. **Flask API**:
   - A simple **Flask application** (`app/app.py`) is created to serve the model. The model is loaded from the serialized file (`best_model.pkl`) and exposed through a REST API. The `/predict_digit` endpoint accepts an image file as input and returns the predicted digit.

3. **Docker Containerization**:
   - The entire application is containerized using **Docker**. The `app/Dockerfile` creates a Docker image containing both the Flask app and the trained model. The container can be deployed on any machine that supports Docker, ensuring portability and scalability.

### Steps for Model Packaging and Deployment:
1. **Training**: Run `src/train.py` to train the model and save it as `models/best_model.pkl`.
2. **Evaluation**: Run `src/evaluate.py` to evaluate the model’s performance, displaying metrics like accuracy and the classification report.
3. **Flask API Deployment**: Run the Flask API server using Docker:
   ```bash
   docker pull DOCKER_USERNAME/mlops-m3:latest
   docker run -p 5000:5000 DOCKER_USERNAME/mlops-m3:latest
   
3. **Test the endpoint**: Send a POST request to the `/predict_digit` endpoint with an image file (e.g., using Postman or curl).
   ```bash
   curl -X POST http://localhost:5000/predict_digit -F "image=@handwritten-digits/digit_1.png"
   ```
    The response will be
```
{
  "prediction": 1
}
```

