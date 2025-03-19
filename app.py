import joblib
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, encoders, and data
model = joblib.load('models/workout_model.pkl')
le_equipment = joblib.load('models/le_equipment.pkl')
le_level = joblib.load('models/le_level.pkl')
le_goal = joblib.load('models/le_goal.pkl')
abdominals_data = pd.read_csv('data/cleaned_abdominals.csv')

# Define form options
equipment_options = abdominals_data['Equipment'].unique().tolist()
level_options = ['Beginner', 'Intermediate', 'Expert']
goal_options = ['Weight Loss', 'Muscle Gain', 'Endurance']

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendation = None
    if request.method == 'POST':
        age = int(request.form['age'])
        weight = float(request.form['weight'])
        equipment = request.form['equipment']
        level = request.form['level']
        goal = request.form['goal']

        # Encode inputs
        equipment_encoded = le_equipment.transform([equipment])[0]
        level_encoded = le_level.transform([level])[0]
        goal_encoded = le_goal.transform([goal])[0]

        # Predict
        input_data = np.array([[age, weight, goal_encoded, equipment_encoded, level_encoded]])
        exercise_title = model.predict(input_data)[0]
        exercise_info = abdominals_data[abdominals_data['Title'] == exercise_title].iloc[0]
        recommendation = f"{exercise_title}: {exercise_info['Desc']}"

    return render_template('index.html', 
                          equipment_options=equipment_options,
                          level_options=level_options,
                          goal_options=goal_options,
                          recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)