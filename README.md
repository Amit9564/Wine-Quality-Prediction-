# Wine-Quality-Prediction-
 Wine Quality Prediction   A machine learning project to predict wine quality based on chemical features like acidity, alcohol, and pH. Includes data preprocessing, model training, and evaluation.


# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Load dataset (CSV file: winequality-red.csv)
# Download from UCI ML repo or Kaggle
data = pd.read_csv("winequality-red.csv")

# 3. Features (X) and Target (y)
X = data.drop("quality", axis=1)
y = data["quality"]

# Optional: Convert quality into binary (Good vs Bad)
y = y.apply(lambda q: 1 if q >= 6 else 0)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train ML Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))