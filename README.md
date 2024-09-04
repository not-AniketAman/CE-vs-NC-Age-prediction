Comparison of Regression Models

This repository contains code for comparing various regression models using a dataset of brain MRI features. The code evaluates the performance of different regression models in predicting age based on MRI features, and also identifies the most important features using SHAP (SHapley Additive exPlanations) values.


Dataset
The dataset used in this project is a collection of MRI features extracted from brain scans. The target variable is the age of the subjects.

Input Features: Various MRI-derived features excluding "Names", "AGE", and "Contrast".
Target Variable: AGE
Categorical Feature: SEX
Models
The following regression models are compared:

Random Forest Regressor
Gradient Boosting Regressor
Support Vector Regressor (SVR)
Linear Regression
Ridge Regression
Lasso Regression
ElasticNet Regression
Feature Importance
SHAP values are used to determine feature importance. For each model, the top 10 most important features are identified and visualized based on the mean absolute SHAP values across all folds of cross-validation.

Dependencies
To run the code, you'll need the following Python libraries:

pandas
numpy
scikit-learn
shap
matplotlib
openpyxl (for reading Excel files)
You can install the necessary dependencies using:

Results
The script performs 5-fold cross-validation and reports the following metrics for each model:

Mean Absolute Error (MAE) and its standard deviation
R2 Score
Top 10 most important features based on SHAP values
Additionally, the feature importance for each model is visualized using bar charts.

Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to create a pull request or open an issue.



