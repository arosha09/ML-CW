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
with open('models/le_exercise_name.pkl', 'rb') as f:
    le_exercise_name = pickle.load(f)

# Load the dataset to get sets and reps
data = pd.read_csv('data/cleaned.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    levels = le_level.classes_
    workout_plan = None

    if request.method == 'POST':
        age = int(request.form['age'])
        weight = int(request.form['weight'])
        level = request.form['level']
        days = int(request.form['days'])

        # Prepare input data
        input_data = pd.DataFrame({
            'Body Part/Muscle': [''],
            'Equipment': [''],
            'Level': [level],
            'Age': [age],
            'Weight': [weight]
        })

        # Encode input data
        input_data['Body Part/Muscle'] = le_body_part.transform(input_data['Body Part/Muscle'])
        input_data['Equipment'] = le_equipment.transform(input_data['Equipment'])
        input_data['Level'] = le_level.transform(input_data['Level'])

        # Generate workout plan
        workout_plan = []
        for day in range(days):
            day_plan = {
                'Day': day + 1,
                'Exercises': []
            }
            for _ in range(6):  # Assuming 6 exercises per day
                workout = model.predict(input_data)
                workout = le_exercise_name.inverse_transform(workout)[0]
                exercise_data = data[data['Exercise Name'] == workout].iloc[0]
                day_plan['Exercises'].append({
                    'Name': workout,
                    'Body_Part': exercise_data['Body Part/Muscle'],
                    'Equipment': exercise_data['Equipment'],
                    'Sets_and_Reps': exercise_data['Sets and Reps']
                })
            workout_plan.append(day_plan)

    return render_template('index.html', levels=levels, workout_plan=workout_plan)

if __name__ == '__main__':
    app.run(debug=True)