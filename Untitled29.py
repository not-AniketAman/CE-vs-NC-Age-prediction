#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt
import sklearn

# Load the data
df = pd.read_excel('/Users/manisha/Desktop/Data Sc Project/Contrast_compile_synth_cat_f.xlsx', sheet_name='Synth_Pre')

# Prepare the data
X = df.drop(["Names", 'AGE', "Contrast"], axis=1)
y = df['AGE']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = ['SEX']

# Check scikit-learn version
sklearn_version = sklearn.__version__
is_old_version = sklearn_version < '1.2'

# Create preprocessing steps
if is_old_version:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
        ])
else:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])

# Define models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf'),
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5)
}

# Function to get SHAP values
def get_shap_values(model, X):
    if hasattr(model, 'predict_proba'):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, X)
    return explainer.shap_values(X)

# Perform 5-Fold Cross-Validation
results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    mae_scores, r2_scores = [], []
    feature_importance = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
        
        # Get SHAP values for this fold
        X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
        fold_shap_values = get_shap_values(pipeline.named_steps['regressor'], X_test_transformed)
        
        # Calculate feature importance for this fold
        if isinstance(fold_shap_values, list):
            fold_importance = np.abs(fold_shap_values).mean(axis=0)
        else:
            fold_importance = np.abs(fold_shap_values).mean(axis=0)
        
        feature_importance.append(fold_importance)
    
    # Calculate mean and standard deviation of metrics
    results[name] = {
        'MAE': np.mean(mae_scores),
        'MAE_std': np.std(mae_scores),
        'R2': np.mean(r2_scores)
    }
    
    # Calculate mean feature importance across all folds
    mean_feature_importance = np.mean(feature_importance, axis=0)
    
    # Get feature names after preprocessing
    feature_names = (numeric_features.tolist() + 
                     pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names_out(categorical_features).tolist())
    
    # Get top 10 features based on mean feature importance
    top_features_idx = mean_feature_importance.argsort()[-10:][::-1]
    top_features = [feature_names[i] for i in top_features_idx]
    
    results[name]['Top_10_Features'] = top_features
    results[name]['Feature_Importance'] = mean_feature_importance

# Print results
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"MAE: {metrics['MAE']:.2f} ± {metrics['MAE_std']:.2f}")
    print(f"R2 Score: {metrics['R2']:.2f}")
    print("Top 10 Features (based on Shapley values):")
    for i, feature in enumerate(metrics['Top_10_Features'], 1):
        importance = metrics['Feature_Importance'][feature_names.index(feature)]
        print(f"{i}. {feature}: {importance:.4f}")

# Visualize feature importance for each model
for name, metrics in results.items():
    plt.figure(figsize=(12, 6))
    importance = [metrics['Feature_Importance'][feature_names.index(f)] for f in metrics['Top_10_Features']]
    y_pos = np.arange(len(metrics['Top_10_Features']))
    plt.barh(y_pos, importance)
    plt.yticks(y_pos, metrics['Top_10_Features'])
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Top 10 Features for {name} (based on Shapley values)')
    plt.tight_layout()
    plt.show()

