import joblib
import numpy as np

# Load model and encoders
model = joblib.load("student_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

def predict_student_performance():
    print("Enter student details for prediction:\n")
    
    gender = input("Gender (Male/Female): ")
    age = int(input("Age: "))
    study_hours_per_day = float(input("Study hours per day: "))
    attendance_percentage = float(input("Attendance percentage: "))
    internal_marks = float(input("Internal marks (out of 40): "))
    previous_sem_cgpa = float(input("Previous semester CGPA: "))
    extra_courses = input("Extra courses (Yes/No): ")
    family_support = input("Family support (Yes/No): ")

    # Encode categorical values
    gender_encoded = label_encoders["gender"].transform([gender])[0]
    extra_courses_encoded = label_encoders["extra_courses"].transform([extra_courses])[0]
    family_support_encoded = label_encoders["family_support"].transform([family_support])[0]

    # Create feature array in correct order
    features = np.array([[gender_encoded, age, study_hours_per_day,
                          attendance_percentage, internal_marks,
                          previous_sem_cgpa, extra_courses_encoded,
                          family_support_encoded]])

    # Predict
    result_encoded = model.predict(features)[0]
    result_label = label_encoders["result"].inverse_transform([result_encoded])[0]

    print(f"\nPredicted Result: {result_label}")

if __name__ == "__main__":
    predict_student_performance()
