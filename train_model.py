import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
df = pd.read_csv("student.csv")

# 2. Encode categorical columns
label_encoders = {}
categorical_cols = ["gender", "extra_courses", "family_support", "result"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 3. Split into features (X) and target (y)
X = df.drop("result", axis=1)
y = df["result"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Model training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save model and encoders
joblib.dump(model, "student_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model and encoders saved successfully.")
