# Step 1: Import Required Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load the Dataset
# Replace path with your actual dataset path
df = pd.read_csv("sample_telco_churn.csv")


# Display first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Step 3: Basic Data Exploration
print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Step 4: Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 5: Handle Missing Values
# Fill numerical columns with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after handling:")
print(df.isnull().sum())

# Step 6: Encode Categorical Variables
label_encoder = LabelEncoder()

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("\nDataset after encoding:")
print(df.head())

# Step 7: Separate Features and Target
# Assuming 'Churn' is the target column
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Step 8: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# Step 9: Feature Scaling (Optional but Recommended)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData Preparation Completed Successfully ✅")
  
