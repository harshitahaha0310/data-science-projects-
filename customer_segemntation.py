# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("sample_Telco_Churn.csv")

# ------------------------------
# Data Preprocessing
# ------------------------------

# Remove customerID if exists
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing values
df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# ------------------------------
# Select Segmentation Features
# ------------------------------
features = df[['tenure', 'MonthlyCharges', 'Contract']]

# Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ------------------------------
# Apply KMeans Clustering
# ------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df['Segment'] = kmeans.fit_predict(scaled_features)

# ------------------------------
# Churn Analysis per Segment
# ------------------------------
print("\nChurn Rate by Segment:")
churn_analysis = df.groupby('Segment')['Churn'].mean()
print(churn_analysis)

# ------------------------------
# Identify High Value Customers
# ------------------------------
high_value = df[
    (df['MonthlyCharges'] > df['MonthlyCharges'].mean()) &
    (df['tenure'] > df['tenure'].mean())
]

print("\nHigh Value Customers At Risk (Churn=1):")
print(high_value[high_value['Churn'] == 1].head())

# ------------------------------
# Visualization
# ------------------------------
plt.figure()
plt.scatter(df['tenure'], df['MonthlyCharges'], c=df['Segment'])
plt.xlabel("Tenure")
plt.ylabel("Monthly Charges")
plt.title("Customer Segmentation")
plt.show()

print("\nTask 3 Completed Successfully ✅")
