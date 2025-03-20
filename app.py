from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and encoders
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/le_body_part.pkl', 'rb') as f:
    le_body_part = pickle.load(f)
with open('models/le_equipment.pkl', 'rb') as f:
    le_equipment = pickle.load(f)
with open('models/le_level.pkl', 'rb') as f:
    le_level = pickle.load(f)

# Load the dataset to get sets and reps
data = pd.read_csv('data/cleaned.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    age = int(request.form['age'])
    weight = float(request.form['weight'])
    fitness_level = request.form['fitness_level']
    target_muscle = request.form['target_muscle']
    equipment = request.form['equipment']

    # Map fitness level to dataset levels
    level_map = {'Beginner': 'Easy', 'Intermediate': 'Intermediate', 'Advanced': 'Hard'}
    level = level_map[fitness_level]

    # Encode inputs
    body_part_encoded = le_body_part.transform([target_muscle])[0]
    equipment_encoded = le_equipment.transform([equipment])[0]
    level_encoded = le_level.transform([level])[0]

    # Predict exercise
    input_data = np.array([[body_part_encoded, equipment_encoded, level_encoded]])
    exercise = model.predict(input_data)[0]

    # Get sets and reps from the dataset
    exercise_info = data[data['Exercise Name'] == exercise].iloc[0]
    sets_reps = exercise_info['Sets and Reps']

    # Generate a simple workout plan
    plan = f"Recommended Exercise: {exercise}\nSets and Reps: {sets_reps}"
    return render_template('result.html', plan=plan)

if __name__ == '__main__':
    app.run(debug=True)