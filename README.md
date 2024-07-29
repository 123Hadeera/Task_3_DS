# Task_3_DS

Heart Disease Prediction with Feature Selection
Overview
This project involves building a classification model to predict heart disease using a dataset from the UCI repository. The dataset is preprocessed, feature-selected, and then used to train and evaluate different machine learning models. The goal is to analyze the importance of features and achieve the best model accuracy.

Dataset
Source: Heart Disease Dataset
Description: The dataset contains information about patients and various attributes related to heart disease. The target variable AHD indicates whether the patient has heart disease (Yes or No).
Project Structure
Data Loading and Preprocessing

Load the dataset from a CSV file.
Drop unnecessary columns, such as 'Unnamed: 0'.
Split the dataset into features (X) and target variable (y).
Identify and separate numeric and categorical features.
Preprocessing Pipelines

Numeric Features:
Impute missing values with mean.
Scale features using MinMaxScaler.
Categorical Features:
Impute missing values with the most frequent value.
One-hot encode categorical features.
Model Pipeline

Combine preprocessing and feature selection steps using a pipeline.
Apply SelectKBest for feature selection based on the chi-squared statistic.
Use RandomForestClassifier, GradientBoostingClassifier, and SVC as models.
Hyperparameter Tuning

Perform Grid Search with cross-validation to find the best hyperparameters for the models.
Model Evaluation

Evaluate the models using accuracy and classification reports.
Analyze feature importances if applicable.
Results
Best Parameters Found: {'feature_selection__k': 12, 'model': RandomForestClassifier(random_state=42), 'model__max_depth': 20, 'model__n_estimators': 50}
Best Score: 0.839
Test Accuracy: 0.758
Classification Report: Provided for detailed performance metrics.
Feature Importance
The feature importances are calculated from the best-performing model. The following table lists the features and their importance scores:

Feature	Importance
Age	...
Sex	...
...	...
Future Work
Experiment with Different Models: Explore models such as Gradient Boosting or SVM to compare performance.
Additional Preprocessing: Try different scaling methods or feature engineering techniques.
Cross-Validation: Implement more rigorous cross-validation strategies.
Advanced Hyperparameter Tuning: Use techniques like RandomizedSearchCV or Bayesian optimization.
Model Interpretation: Utilize tools like SHAP or LIME for better model interpretation.
Installation and Usage
Install Required Libraries:

bash
Copy code
pip install pandas numpy scikit-learn
Run the Code:

Ensure that the dataset is available at the specified path.
Execute the script to load, preprocess, and model the data.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
UCI Machine Learning Repository
Kaggle Dataset
