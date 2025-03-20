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
    levels = ['easy', 'intermediate', 'hard']
    workout_plan = None

    if request.method == 'POST':
        age = int(request.form['age'])
        weight = int(request.form['weight'])
        level = request.form['level']
        days = int(request.form['days'])
        include_core = 'include_core' in request.form

        # Define muscle groups for each day
        muscle_groups = {
            2: [['Chest', 'arms', 'Shoulders'], ['Back', 'legs']],
            3: [['chest', 'shoulders', 'triceps'], ['back', 'biceps'], ['legs']]
        }

        # Use a valid value from the training data for 'Equipment'
        valid_equipment = data['Equipment'].mode()[0]

        # Generate workout plan
        workout_plan = []
        for day in range(days):
            day_plan = {
                'Day': day + 1,
                'Exercises': []
            }
            for muscle_group in muscle_groups[days][day]:
                muscle_data = data[data['Body Part/Muscle'].str.contains(muscle_group, case=False, na=False)]
                for _ in range(3):  # Assuming 3 exercises per muscle group per day
                    if not muscle_data.empty:
                        # Prepare input data for prediction
                        input_data = pd.DataFrame({
                            'Body Part/Muscle': [muscle_group],
                            'Equipment': [valid_equipment],
                            'Level': [level],
                            'Age': [age],
                            'Weight': [weight]
                        })

                        # Encode input data
                        input_data['Body Part/Muscle'] = le_body_part.transform(input_data['Body Part/Muscle'])
                        input_data['Equipment'] = le_equipment.transform(input_data['Equipment'])
                        input_data['Level'] = le_level.transform(input_data['Level'])

                        # Predict exercise
                        workout = model.predict(input_data)
                        workout = le_exercise_name.inverse_transform(workout)[0]
                        exercise_data = muscle_data[muscle_data['Exercise Name'] == workout]
                        if not exercise_data.empty:
                            exercise_data = exercise_data.iloc[0]
                            day_plan['Exercises'].append({
                                'Name': workout,
                                'Body_Part': exercise_data['Body Part/Muscle'],
                                'Equipment': exercise_data['Equipment'],
                                'Sets_and_Reps': exercise_data['Sets and Reps']
                            })
            if include_core:
                core_data = data[data['Body Part/Muscle'].str.contains('core', case=False, na=False)]
                for _ in range(3):  # Assuming 3 core exercises per day
                    if not core_data.empty:
                        # Prepare input data for prediction
                        input_data = pd.DataFrame({
                            'Body Part/Muscle': ['core'],
                            'Equipment': [valid_equipment],
                            'Level': [level],
                            'Age': [age],
                            'Weight': [weight]
                        })

                        # Encode input data
                        input_data['Body Part/Muscle'] = le_body_part.transform(input_data['Body Part/Muscle'])
                        input_data['Equipment'] = le_equipment.transform(input_data['Equipment'])
                        input_data['Level'] = le_level.transform(input_data['Level'])

                        # Predict exercise
                        workout = model.predict(input_data)
                        workout = le_exercise_name.inverse_transform(workout)[0]
                        exercise_data = core_data[core_data['Exercise Name'] == workout]
                        if not exercise_data.empty:
                            exercise_data = exercise_data.iloc[0]
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