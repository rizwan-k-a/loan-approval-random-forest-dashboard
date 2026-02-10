# Build a complete Machine Learning pipeline for Loan Approval Prediction.
# Use pandas, seaborn, matplotlib, scikit-learn.
# Do EDA, preprocessing, encoding, train-test split.
# Train RandomForestClassifier.
# Show accuracy, confusion matrix, precision, recall.
# Create plots: histogram, countplot, boxplot, heatmap.
# Dataset file name: loan_approval_dataset.csv
# Target column: Loan_Status

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("loan_approval_dataset.csv")

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Clean string values in all columns
for col in df.columns:
    if df[col].dtype == "object" or df[col].dtype == "str":
        df[col] = df[col].astype(str).str.strip()

print("\nShape:", df.shape)
print("\nFirst 5 Records (HEAD):")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nSummary:")
print(df.describe())

# -------------------------
# EDA (Q2–Q3 REQUIREMENTS)
# -------------------------

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# No missing values in this dataset, so no need to fill

# Histograms
df.hist(figsize=(10,8))
plt.tight_layout()
plt.show()

# Count plot target
sns.countplot(x="loan_status", data=df)
plt.title("Loan Status Count")
plt.show()

# Boxplot
if "loan_amount" in df.columns:
    sns.boxplot(y=df["loan_amount"])
    plt.title("Loan Amount Boxplot")
    plt.show()

# Correlation heatmap (numeric only)
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------
# ENCODING
# -------------------------
le_dict = {}

# Get non-numeric columns
non_numeric_cols = df.select_dtypes(exclude=['number']).columns

for col in non_numeric_cols:
    le = LabelEncoder()
    encoded_values = le.fit_transform(df[col].astype(str))
    df[col] = encoded_values  # Assign encoded integer values
    le_dict[col] = le

print("\nAfter encoding:")
print(df.dtypes)

# -------------------------
# SPLIT DATA
# -------------------------
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# MODEL — RANDOM FOREST
# -------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# PREDICTION
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# EVALUATION
# -------------------------
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred))

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:\n", importances)

importances.plot(kind="bar")
plt.title("Feature Importance")
plt.show()
