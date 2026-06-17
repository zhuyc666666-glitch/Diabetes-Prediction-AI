# MedPredict AI: Diabetes Risk Prediction System

## 1. Project Overview

MedPredict AI is a machine learning project designed to estimate diabetes risk based on demographic information, clinical indicators, and lifestyle-related features. The project combines exploratory data analysis, supervised machine learning model comparison, model interpretation, and a Streamlit-based web application.

This project is intended as an undergraduate medical AI demonstration project. It shows how basic clinical data can be transformed into a predictive workflow that supports risk awareness and educational decision support.

## 2. Dataset Description

The project uses the `diabetes_prediction_dataset.csv` dataset. The dataset contains patient-level records with demographic variables, medical history indicators, laboratory-related measurements, and a binary diabetes outcome.

The target variable is:

- `diabetes`: binary label indicating whether the patient has diabetes.

Before model training, duplicate records are removed to improve data quality and reduce repeated-sample bias.

## 3. Features Used

The prediction model uses the following input features:

- `gender`
- `age`
- `hypertension`
- `heart_disease`
- `smoking_history`
- `bmi`
- `HbA1c_level`
- `blood_glucose_level`

Categorical variables such as `gender` and `smoking_history` are processed using One-Hot Encoding. Numerical and binary clinical variables are used as model inputs after preprocessing.

## 4. Machine Learning Models

The training pipeline compares three supervised classification models:

- Logistic Regression
- Random Forest
- Gradient Boosting

Each model is trained on the same train-test split. The model with the highest ROC-AUC score is automatically selected as the best model and saved for use in the web application.

For interpretability, tree-based feature importance analysis is also included. If the best model supports `feature_importances_`, its feature importance values are used. If not, the Random Forest model is used as an interpretable tree-based reference model.

## 5. Model Evaluation Metrics

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification Report

The project also generates visual evaluation outputs:

- Confusion Matrix
- ROC Curve
- Feature Importance Plot

These outputs help assess model discrimination, classification performance, and the relative contribution of each feature.

## 6. Web Application

The project includes a Streamlit web application for diabetes risk prediction. Users can enter patient-related information, including demographic features, clinical history, BMI, HbA1c level, and blood glucose level.

The application outputs:

- Diabetes probability
- Prediction result: Low Risk, Medium Risk, or High Risk
- Top 5 Risk Factors based on feature importance analysis

Risk levels are defined as:

- Probability `< 0.3`: Low Risk
- Probability `0.3 - 0.7`: Medium Risk
- Probability `>= 0.7`: High Risk

## 7. How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```text
data/diabetes_prediction_dataset.csv
```

Run exploratory data analysis:

```bash
python eda.py
```

Train and compare machine learning models:

```bash
python train.py
```

Start the Streamlit web application:

```bash
streamlit run app.py
```

## 8. Project Structure

```text
Diabetes-Prediction-AI/
|-- data/
|   `-- diabetes_prediction_dataset.csv
|-- figures/
|   |-- age_distribution.png
|   |-- bmi_distribution.png
|   |-- hba1c_distribution.png
|   |-- blood_glucose_distribution.png
|   |-- diabetes_count.png
|   |-- confusion_matrix.png
|   |-- roc_curve.png
|   `-- feature_importance.png
|-- models/
|   |-- best_diabetes_model.pkl
|   |-- feature_columns.pkl
|   `-- feature_importance.csv
|-- eda.py
|-- train.py
|-- app.py
|-- requirements.txt
`-- README.md
```

## 9. Disclaimer

This project is for educational and research demonstration purposes only. It is not intended to provide medical diagnosis, treatment recommendations, or clinical decision-making guidance.

The prediction results should not be used as a substitute for professional medical evaluation. Patients should consult qualified healthcare professionals for diagnosis, risk assessment, and treatment planning.

## 10. Future Improvements

Possible future improvements include:

- Adding model calibration to improve probability reliability.
- Testing additional models such as XGBoost or LightGBM.
- Performing cross-validation for more robust evaluation.
- Adding SHAP-based explanations for individual predictions.
- Improving data validation and handling missing or abnormal clinical values.
- Deploying the application to a cloud platform for easier access.
- Expanding the interface with patient-friendly visual explanations.
