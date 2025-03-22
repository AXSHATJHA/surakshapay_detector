import pandas as pd
import numpy as np
import joblib
import json

# Load the trained model
model = joblib.load("fraud_detection.pkl")

# Load test data from a JSON file (preserve original data as a list of dicts)
test_file_path = "test_transactions.json"
with open(test_file_path, "r") as file:
    test_data = json.load(file)  # test_data is a list of dictionaries

# Create a DataFrame for processing while keeping the original data intact
df_test = pd.DataFrame(test_data)

# Handle missing numerical values
numerical_columns = ['transaction_amount', 'hour', 'minute', 'txn_count_1h', 'recent_txn_density']
for col in numerical_columns:
    if col in df_test.columns:
        df_test[col] = df_test[col].fillna(df_test[col].median())

# Convert transaction_date to timestamp if present (without dropping it)
if 'transaction_date' in df_test.columns:
    df_test['transaction_date_processed'] = pd.to_datetime(df_test['transaction_date'], errors='coerce')
    df_test['transaction_timestamp'] = df_test['transaction_date_processed'].astype(int) // 10**9
    df_test.drop(columns=['transaction_date_processed'], inplace=True)

# Define categorical columns and convert to integer encoding
categorical_columns = [
    'transaction_channel', 'payer_email_anonymous', 'payee_ip_anonymous',
    'transaction_payment_mode_anonymous', 'payment_gateway_bank_anonymous',
    'payer_browser_anonymous', 'transaction_id_anonymous', 'payee_id_anonymous'
]
for col in categorical_columns:
    if col in df_test.columns:
        df_test[col] = df_test[col].astype(str).factorize()[0]

# Align test data with the model's expected features
trained_feature_names = model.booster_.feature_name()
for feature in trained_feature_names:
    if feature not in df_test.columns:
        df_test[feature] = 0
df_test = df_test[trained_feature_names]

# Predict fraud labels (0 or 1)
predictions = model.predict(df_test)

# Add predictions back to the original test_data
for i, transaction in enumerate(test_data):
    transaction['fraud'] = int(predictions[i])

# Save the updated test_data with fraud predictions as JSON
output_file_path = "fraud_detection_results.json"
with open(output_file_path, "w") as file:
    json.dump(test_data, file, indent=4)

print(f"Fraud predictions saved to {output_file_path}")