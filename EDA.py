# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# 2. Load Dataset
df = pd.read_csv("sample_telco_churn.csv")

# 3. Basic Data Overview
print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# 4. Overall Churn Rate Calculation
churn_rate = df['Churn'].value_counts(normalize=True) * 100
print("\nOverall Churn Rate (%):")
print(churn_rate)

# 5. Visualize Overall Churn Rate
plt.figure()
sns.countplot(x='Churn', data=df)
plt.title("Overall Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

# 6. Churn Distribution by Gender
plt.figure()
sns.countplot(x='gender', hue='Churn', data=df)
plt.title("Churn Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.show()

# 7. Churn Distribution by Partner Status
plt.figure()
sns.countplot(x='Partner', hue='Churn', data=df)
plt.title("Churn Distribution by Partner Status")
plt.xlabel("Partner")
plt.ylabel("Number of Customers")
plt.show()

# 8. Churn Distribution by Dependent Status
plt.figure()
sns.countplot(x='Dependents', hue='Churn', data=df)
plt.title("Churn Distribution by Dependent Status")
plt.xlabel("Dependents")
plt.ylabel("Number of Customers")
plt.show()

# 9. Tenure Distribution
plt.figure()
sns.histplot(df['tenure'], bins=30)
plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Customer Count")
plt.show()

# 10. Tenure vs Churn
plt.figure()
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title("Tenure vs Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.show()

# 11. Churn by Contract Type
plt.figure()
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.show()

# 12. Churn by Payment Method
plt.figure()
sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.xticks(rotation=45)
plt.title("Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.show()

print("\nEDA Task 2 Completed Successfully!")
