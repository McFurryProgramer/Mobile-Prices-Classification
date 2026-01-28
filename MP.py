import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.head())
print(train_df.info())
print(train_df.isna().sum())

plt.figure(figsize=(6,4))
sns.countplot(x='price_range', data=train_df,hue='price_range', palette='viridis', legend=False)
plt.title('Распределение классов price_range')
plt.xlabel('Price Range')
plt.ylabel('Количество')
plt.show()

plt.figure(figsize=(12,10))
corr = train_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Корреляция признаков')
plt.show()

X = train_df.drop("price_range", axis=1)
y = train_df["price_range"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_df.drop("id", axis=1))

# 4. Логистическая регрессия

log_reg = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)
log_reg.fit(X_train_scaled, y_train)

y_val_pred_lr = log_reg.predict(X_val_scaled)
print("=== Logistic Regression (Validation) ===")
print(classification_report(y_val, y_val_pred_lr))

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_val, y_val_pred_lr), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression (Validation)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 5. Random Forest

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_val_pred_rf = rf_clf.predict(X_val)

print("=== Random Forest (Validation) ===")
print(classification_report(y_val, y_val_pred_rf))

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_val, y_val_pred_rf), annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Random Forest (Validation)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

importances = rf_clf.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', hue='Feature',data=feat_df, palette='viridis')
plt.title('Важность признаков - Random Forest')
plt.show()

y_test_pred_lr = log_reg.predict(X_test_scaled)
y_test_pred_rf = rf_clf.predict(test_df.drop("id", axis=1))



