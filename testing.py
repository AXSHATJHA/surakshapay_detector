import pandas as pd
import numpy as np
import joblib
import json

# Load the trained model
model = joblib.load("fraud_detection.pkl")

# Load test data from a JSON file
test_file_path = "test_transactions.json"
with open(test_file_path, "r") as file:
    test_data = json.load(file)

df_test = pd.DataFrame(test_data)

# Handle missing numerical values
numerical_columns = ['transaction_amount', 'hour', 'minute', 'txn_count_1h', 'recent_txn_density']
for col in numerical_columns:
    if col in df_test.columns:
        df_test[col] = df_test[col].fillna(df_test[col].median())

# Convert transaction_date to timestamp if present
if 'transaction_date' in df_test.columns:
    df_test['transaction_date'] = pd.to_datetime(df_test['transaction_date'], errors='coerce')
    df_test['transaction_timestamp'] = df_test['transaction_date'].astype(int) // 10**9
    df_test.drop(columns=['transaction_date'], inplace=True)

# Define categorical columns (ensuring consistency with training data)
categorical_columns = [
    'transaction_channel', 'payer_email_anonymous', 'payee_ip_anonymous',
    'transaction_payment_mode_anonymous', 'payment_gateway_bank_anonymous',
    'payer_browser_anonymous', 'transaction_id_anonymous', 'payee_id_anonymous'
]

# Convert categorical columns to integer encoding
for col in categorical_columns:
    if col in df_test.columns:
        df_test[col] = df_test[col].astype(str).factorize()[0]

# Load feature names from trained model to match exact order
trained_feature_names = model.booster_.feature_name()

# Ensure test data has all required features
for feature in trained_feature_names:
    if feature not in df_test.columns:
        df_test[feature] = 0  # Assign default value 0 for missing features

# Align test data columns to match training order
df_test = df_test[trained_feature_names]

# Predict fraud probabilities
predictions = model.predict(df_test)
df_test['fraud_prediction'] = predictions

# Save results
df_test.to_csv("fraud_detection_results.csv", index=False)
print("Fraud predictions saved to fraud_detection_results.csv")
