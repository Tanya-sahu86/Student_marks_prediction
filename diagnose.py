import pandas as pd
import joblib
import numpy as np

# Load data
try:
    df = pd.read_csv("student.csv")
except Exception as e:
    print("Failed to read student.csv:", e)
    raise

print("Result distribution:\n", df['result'].value_counts(dropna=False))

# Load model and encoders
model = joblib.load("student_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

print("Model classes:", getattr(model, 'classes_', None))
print("Feature names used in training (if available):", getattr(model, 'feature_names_in_', None))

# Build a sample input similar to what you entered interactively
sample_input = {
    'gender': 'Male',
    'age': 19,
    'study_hours_per_day': 2.0,
    'attendance_percentage': 75.0,
    'internal_marks': 35.0,
    'previous_sem_cgpa': 8.0,
    'extra_courses': 'Yes',
    'family_support': 'Yes'
}

# Encode categorical fields
def enc(col, val):
    le = label_encoders[col]
    return le.transform([val])[0]

sample = np.array([[
    enc('gender', sample_input['gender']),
    sample_input['age'],
    sample_input['study_hours_per_day'],
    sample_input['attendance_percentage'],
    sample_input['internal_marks'],
    sample_input['previous_sem_cgpa'],
    enc('extra_courses', sample_input['extra_courses']),
    enc('family_support', sample_input['family_support'])
]])

print("Sample array:", sample)

pred = model.predict(sample)
print("Raw model prediction (encoded):", pred)

# If model supports predict_proba, show probabilities
if hasattr(model, 'predict_proba'):
    print("Prediction probabilities:", model.predict_proba(sample))

# Inverse-transform predicted label if possible
if 'result' in label_encoders:
    print("Predicted label:", label_encoders['result'].inverse_transform(pred))
else:
    print("No 'result' encoder found to inverse-transform.")
