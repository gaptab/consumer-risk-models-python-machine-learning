# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
# data creation
np.random.seed(42)
data = pd.DataFrame({
    "Age": np.random.randint(18, 70, 1000),
    "Income": np.random.randint(20000, 150000, 1000),
    "LoanAmount": np.random.randint(5000, 50000, 1000),
    "CreditScore": np.random.randint(300, 850, 1000),
    "Default": np.random.choice([0, 1], size=1000, p=[0.8, 0.2])
})

# Data preparation
X = data.drop("Default", axis=1)
y = data["Default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Validation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, y_prob))

def validate_model(model, X_test, y_test):
    """
    Standardized model validation function.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "AUC-ROC": roc_auc_score(y_test, y_prob),
        "Classification Report": classification_report(y_test, y_pred, output_dict=True)
    }
    return metrics

# Example usage
metrics = validate_model(model, X_test, y_test)
print("Standardized Metrics:\n", metrics)


def generate_mrm_guidelines():
    """
    Generates MRM guidelines for machine learning models.
    """
    guidelines = """
    Model Risk Management Guidelines:
    1. Define clear roles for model development, validation, and usage.
    2. Ensure interpretability and explainability of machine learning models.
    3. Validate models against regulatory and business standards.
    4. Periodically monitor and recalibrate models to maintain performance.
    """
    return guidelines

def write_white_paper(title, content):
    """
    Simulates writing a white paper by saving content to a file.
    """
    with open(f"{title}.txt", "w") as file:
        file.write(content)

# Generate guidelines and white paper
guidelines = generate_mrm_guidelines()
write_white_paper("MRM_Guidelines", guidelines)
print("MRM guidelines saved in 'MRM_Guidelines.txt'")

def gap_assessment(expected_metrics, actual_metrics):
    """
    Compares expected vs actual model performance metrics.
    """
    gaps = {}
    for metric in expected_metrics.keys():
        gap = actual_metrics.get(metric, 0) - expected_metrics[metric]
        gaps[metric] = gap
    return gaps

# Example
expected = {"AUC-ROC": 0.85}
actual = {"AUC-ROC": metrics["AUC-ROC"]}
gaps = gap_assessment(expected, actual)
print("Validation Gaps:\n", gaps)



# Visualizing the class distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x="Default", palette="Set2")
plt.title("Default Class Distribution")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Feature importance visualization
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importances, x="Importance", y="Feature", palette="Set3")
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# ROC Curve visualization


fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# Save data to CSV
data.to_csv("consumer_risk_data.csv", index=False)
print("Dummy data saved to 'consumer_risk_data.csv'.")

# Save feature importance to CSV
feature_importances.to_csv("feature_importances.csv", index=False)
print("Feature importances saved to 'feature_importances.csv'.")
