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

# Load the dataset
data = pd.read_csv('data/cleaned.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    levels = ['easy', 'intermediate', 'hard']
    workout_plan = None

    if request.method == 'POST':
        age = int(request.form['age'])
        weight = int(request.form['weight'])
        level_input = request.form['level'].lower()
        days = int(request.form['days'])
        include_core = 'include_core' in request.form

        placeholder_exercise = data.iloc[0]
        
        sample = np.array([[
            le_body_part.transform([placeholder_exercise['Body Part/Muscle']])[0],
            le_equipment.transform([placeholder_exercise['Equipment']])[0],
            3,  
            10, 
            12, 
            1,  
            0,  
            1   
        ]])
        
        predicted_level_encoded = model.predict(sample)[0]
        predicted_level = le_level.inverse_transform([predicted_level_encoded])[0]

        # Use predicted level if no user input, otherwise use user-provided level
        level = predicted_level if level_input not in levels else level_input

        # Define muscle groups for each day
        muscle_groups = {
            2: [['Chest', 'Traps', 'Shoulders', 'Biceps', 'Triceps'], 
                ['Lower Back', 'Back', 'Quads', 'Hamstrings', 'Glutes', 'Calves', 'Inner Thighs']],
            3: [['Chest', 'Shoulders', 'Triceps'], 
                ['Back', 'Lower Back', 'Traps', 'Biceps'], 
                ['Quads', 'Hamstrings', 'Glutes', 'Calves', 'Inner Thighs']]
        }

        # Generate workout plan
        workout_plan = []
        used_exercises = set()
        for day in range(min(days, len(muscle_groups.get(days, muscle_groups[2])))):
            day_plan = {'Day': day + 1, 'Exercises': {}, 'Core_Exercises': []}
            for muscle_group in muscle_groups[days][day]:
                muscle_data = data[
                    (data['Body Part/Muscle'].str.contains(muscle_group, case=False, na=False)) &
                    (data['Level'].str.lower() == level) &
                    (~data['Exercise Name'].isin(used_exercises))
                ]
                num_exercises = np.random.randint(3, 5)
                selected_exercises = muscle_data.sample(n=min(num_exercises, len(muscle_data)), replace=False)
                day_plan['Exercises'][muscle_group] = []
                for _, exercise_data in selected_exercises.iterrows():
                    day_plan['Exercises'][muscle_group].append({
                        'Name': exercise_data['Exercise Name'],
                        'Body_Part': exercise_data['Body Part/Muscle'],
                        'Equipment': exercise_data['Equipment'],
                        'Sets_and_Reps': exercise_data['Sets and Reps']
                    })
                    used_exercises.add(exercise_data['Exercise Name'])
            if include_core:
                core_data = data[
                    (data['Body Part/Muscle'].str.contains('Core', case=False, na=False)) &
                    (data['Level'].str.lower() == level) &
                    (~data['Exercise Name'].isin(used_exercises))
                ]
                num_core_exercises = np.random.randint(3, 5)
                selected_core_exercises = core_data.sample(n=min(num_core_exercises, len(core_data)), replace=False)
                for _, exercise_data in selected_core_exercises.iterrows():
                    day_plan['Core_Exercises'].append({
                        'Name': exercise_data['Exercise Name'],
                        'Body_Part': exercise_data['Body Part/Muscle'],
                        'Equipment': exercise_data['Equipment'],
                        'Sets_and_Reps': exercise_data['Sets and Reps']
                    })
                    used_exercises.add(exercise_data['Exercise Name'])
            workout_plan.append(day_plan)

    return render_template('index.html', levels=levels, workout_plan=workout_plan)

if __name__ == '__main__':
    app.run(debug=True)