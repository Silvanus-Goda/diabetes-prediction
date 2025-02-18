import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

# Load dataset
pima = fetch_openml(name='diabetes', version=1, as_frame=True)
df = pima.frame

# Rename columns
df.columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df['Outcome'] = df['Outcome'].map({'tested_positive': 1, 'tested_negative': 0})

# Handle missing values
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
for col in df.columns[:-1]:  
    df[col] = df[col].fillna(df[col].median())

# Split data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
decision = DecisionTreeClassifier(max_depth=5)
svm = SVC(kernel='linear', probability=True)
ann = MLPClassifier(hidden_layer_sizes=(5,2), max_iter=2000, learning_rate_init=0.0001)
rf = RandomForestClassifier(n_estimators=100)

# Ensemble Model
ensemble = VotingClassifier(estimators=[
    ('dt', decision),
    ('svm', svm),
    ('ann', ann),
    ('rf', rf)
], voting='hard')

ensemble.fit(X_train, y_train)

# Save trained model
joblib.dump(ensemble, "diabetes_model.pkl")
print("✅ Model saved as 'diabetes_model.pkl'")
