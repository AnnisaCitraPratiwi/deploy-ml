from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('classification_diagnose.h5')

# Mapping for predictions
inverse_severity_mapping = {
    0: 'Generalized Anxiety',
    1: 'Panic Disorder',
    2: 'Major Depressive Disorder',
    3: 'Bipolar Disorder'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    age = float(request.form['age'])
    gender = float(request.form['gender'])
    symptom_severity = float(request.form['symptom_severity'])
    mood_score = float(request.form['mood_score'])
    sleep_quality = float(request.form['sleep_quality'])
    physical_activity = float(request.form['physical_activity'])
    stress_level = float(request.form['stress_level'])

    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Symptom Severity (1-10)': [symptom_severity],
        'Mood Score (1-10)': [mood_score],
        'Sleep Quality (1-10)': [sleep_quality],
        'Physical Activity (hrs/week)': [physical_activity],
        'Stress Level (1-10)': [stress_level]
    })

    # Predict
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_diagnosis = inverse_severity_mapping[predicted_class]

    return render_template('index.html', prediction=predicted_diagnosis)

if __name__ == '__main__':
    app.run(debug=True)
