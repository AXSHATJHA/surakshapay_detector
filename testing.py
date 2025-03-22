# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any, Optional

# Load the trained model
model = joblib.load("fraud_detection (1).pkl")

# Define the input schema for the API
class Transaction(BaseModel):
    transaction_amount: float
    transaction_date: str
    transaction_channel: str
    payer_email_anonymous: str
    payee_ip_anonymous: Optional[str] = None  # Make this field optional
    transaction_id_anonymous: str
    payee_id_anonymous: str

class TransactionBatch(BaseModel):
    transactions: List[Transaction]

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict", response_model=List[Dict[str, Any]])
async def predict_fraud(batch: TransactionBatch):
    try:
        # Convert input data to DataFrame
        transactions = [t.dict() for t in batch.transactions]
        df_test = pd.DataFrame(transactions)

        # Handle missing numerical values
        numerical_columns = ['transaction_amount', 'hour', 'minute', 'txn_count_1h', 'recent_txn_density']
        for col in numerical_columns:
            if col in df_test.columns:
                df_test[col] = df_test[col].fillna(df_test[col].median())

        # Convert transaction_date to timestamp if present
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

        # Add predictions back to the original transactions
        for i, transaction in enumerate(transactions):
            transaction['fraud'] = int(predictions[i])

        return transactions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))