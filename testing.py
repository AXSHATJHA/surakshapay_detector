import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv("transactions_train.csv")
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# ================== Filter Transactions with Amount < 10 ==================
df = df[df['transaction_amount'] >= 10]  # Remove transactions with amount less than 10

# ================== Advanced Feature Engineering ==================
# Time-based patterns
df['hour'] = df['transaction_date'].dt.hour
df['is_night'] = df['hour'].between(0, 5).astype(int)
df['minute'] = df['transaction_date'].dt.minute

# Payment behavior patterns
df['payment_risk'] = np.where(df['transaction_payment_mode_anonymous'].isin([10, 11]), 2,
                             np.where(df['transaction_payment_mode_anonymous'] == 6, 1, 0))

# Amount anomalies
df['amount_log'] = np.log1p(df['transaction_amount'])
df['amount_diff'] = df.groupby('payer_email_anonymous')['transaction_amount'].diff().fillna(0)

# Frequency patterns with time decay
df = df.sort_values(['payer_email_anonymous', 'transaction_date'])
df['txn_count_1h'] = df.groupby('payer_email_anonymous')['transaction_date'].transform(
    lambda x: x.diff().dt.total_seconds().lt(3600).cumsum()
)
df['recent_txn_density'] = df['txn_count_1h'] / (df['hour'] + 1)

# Network patterns
df['unique_payees_per_payer'] = df.groupby('payer_email_anonymous')['payee_ip_anonymous'].transform('nunique')
df['payer_payee_combinations'] = df.groupby(['payer_email_anonymous', 'payee_ip_anonymous']).transform('size')

# ================== Remove Rows with NaN Values ==================
df = df.dropna()  # Remove all rows with NaN values

# ================== Advanced Anomaly Detection ==================
# Isolation Forest
iso = IsolationForest(contamination=0.2, random_state=42)
df['iso_anomaly'] = iso.fit_predict(df[['transaction_amount', 'hour', 'minute']])

# Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.2)
df['lof_anomaly'] = lof.fit_predict(df[['transaction_amount', 'hour', 'minute']])

# ================== Feature Selection ==================
features = [
    'transaction_amount', 'amount_log', 'amount_diff',
    'txn_count_1h', 'recent_txn_density', 'payment_risk',
    'unique_payees_per_payer', 'payer_payee_combinations',
    'iso_anomaly', 'lof_anomaly', 'is_night', 'hour', 'minute'
]

# ================== Data Preparation ==================
X = df[features]
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ================== Advanced Resampling ==================
# Combine SMOTE and RandomUnderSampler
resampling_pipeline = Pipeline([
    ('oversample', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=2)),
    ('undersample', RandomUnderSampler(sampling_strategy=0.5, random_state=42))
])
X_res, y_res = resampling_pipeline.fit_resample(X_train, y_train)

# ================== Model Training ==================
# Use LightGBM with focal loss for imbalanced data
model = LGBMClassifier(
    n_estimators=500,
    class_weight='balanced',
    learning_rate=0.01,
    max_depth=7,
    objective='binary',
    random_state=42
)
model.fit(X_res, y_res)

# ================== Fixed Threshold = 0.5 ==================
threshold = 0.01  # Explicitly set threshold to 0.5
y_pred = (model.predict_proba(X_test)[:, 1] > threshold).astype(int)

# ================== Evaluation ==================
print("Final Threshold:", threshold)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))

precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
print(f"Precision-Recall AUC: {auc(recall, precision):.4f}")

# ================== Fraud Pattern Analysis ==================
print("\nTop Fraud Indicators:")
fraud_data = df[df['is_fraud'] == 1]
print(f"1. Night Transactions: {fraud_data['is_night'].mean():.0%}")
print(f"2. Payment Risk Score 2: {fraud_data[fraud_data['payment_risk'] == 2].shape[0]/len(fraud_data):.0%}")
print(f"3. High Recent Density (>5): {fraud_data[fraud_data['recent_txn_density'] > 5].shape[0]/len(fraud_data):.0%}")
print(f"4. ISO Anomaly Flagged: {fraud_data[fraud_data['iso_anomaly'] == -1].shape[0]/len(fraud_data):.0%}")
print(f"5. LOF Anomaly Flagged: {fraud_data[fraud_data['lof_anomaly'] == -1].shape[0]/len(fraud_data):.0%}")