import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load dataset (replace 'bank_churn.csv' with your file path)
try:
    df = pd.read_csv('bank_churn.csv')
except FileNotFoundError:
    print("Error: 'bank_churn.csv' not found. Please ensure the dataset file is in the correct directory.")
    exit()

# Drop customer_id (not predictive)
df = df.drop('customer_id', axis=1)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['country', 'gender'], drop_first=True)

# Define features (X) and target (y)
X = df.drop('churn', axis=1)
y = df['churn']

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Evaluate model
from sklearn.metrics import roc_auc_score, classification_report
y_pred = model.predict(X_test)
print("Model Evaluation:")
print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))