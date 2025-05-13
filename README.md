# Breast Cancer Classifier

## Project Overview

This project aims to classify breast tumors as **benign** or **malignant** using multiple machine learning models. The **Breast Cancer Wisconsin Diagnostic dataset** is used for training and evaluation. Three different algorithms were implemented and compared:
- **Support Vector Machine (SVM)** with hyperparameter tuning
- **Neural Network (Multi-layer Perceptron)**
- **Logistic Regression**

Model performance was evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

## Design Overview

### Data Preprocessing
- **Dataset**: Breast Cancer Wisconsin Diagnostic dataset from `sklearn.datasets`.
- **Features**: Includes cell characteristics such as radius, texture, perimeter, area, and smoothness.
- **Target Labels**:
  - `0` = Benign
  - `1` = Malignant
- **Steps**:
  - Loaded the dataset into a pandas DataFrame.
  - Checked and confirmed absence of missing values.
  - Standardized feature values using `StandardScaler`.
  - Split into 80% training and 20% testing sets.

### Models Used
1. **Support Vector Machine (SVM)**
   - Hyperparameter tuning via **GridSearchCV**
   - Best parameters: `{'C': 1, 'gamma': 'scale', 'kernel': 'linear'}`
2. **Neural Network (MLPClassifier)**
   - Configured with two hidden layers and ReLU activation
   - Optimizer: Adam, with early stopping and fixed random state
3. **Logistic Regression**
   - Used as a baseline for comparison

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix (visualized using Seaborn)

### Visualization
- **Confusion Matrix Heatmaps**: For each model to assess classification performance
- **Classification Report**: For numerical comparison of precision, recall, and F1-score

## Key Components

### Libraries Used
- `pandas`, `numpy`: Data handling and numerical computation
- `matplotlib`, `seaborn`: Visualizations
- `sklearn`: Data loading, preprocessing, models, evaluation metrics
- `GridSearchCV`: To tune SVM hyperparameters
- `MLPClassifier`: Neural Network from `sklearn.neural_network`

### Workflow
1. Load and explore the dataset
2. Preprocess and standardize the features
3. Split into training and testing data
4. Train SVM using GridSearchCV to find optimal parameters
5. Train Neural Network with MLPClassifier
6. Train Logistic Regression for baseline comparison
7. Evaluate all models
8. Visualize results using confusion matrices and classification reports

## Conclusion

This project demonstrates the power of machine learning models — especially **SVMs and Neural Networks** — in medical diagnosis tasks. With accuracy scores above 97%, these models are promising tools for supporting cancer diagnosis decisions.
