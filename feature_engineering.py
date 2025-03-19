import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Load cleaned data
abdominals_data = pd.read_csv('data/cleaned_abdominals.csv')

# Simulate user data
np.random.seed(42)
n_samples = 1000
user_data = pd.DataFrame({
    'Age': np.random.randint(18, 60, n_samples),
    'Weight': np.random.uniform(50, 120, n_samples),
    'Goal': np.random.choice(['Weight Loss', 'Muscle Gain', 'Endurance'], n_samples)
})
user_exercises = user_data.copy()
user_exercises['Exercise'] = np.random.choice(abdominals_data['Title'], n_samples)

# Merge with exercise data
dataset = user_exercises.merge(abdominals_data, left_on='Exercise', right_on='Title', how='left')

# Encode categorical variables
le_equipment = LabelEncoder()
le_level = LabelEncoder()
le_goal = LabelEncoder()

dataset['Equipment'] = le_equipment.fit_transform(dataset['Equipment'])
dataset['Level'] = le_level.fit_transform(dataset['Level'])
dataset['Goal'] = le_goal.fit_transform(dataset['Goal'])

# Define features and target
X = dataset[['Age', 'Weight', 'Goal', 'Equipment', 'Level']]
y = dataset['Title']

# Save processed data and encoders
dataset.to_csv('data/processed_data.csv', index=False)
joblib.dump(le_equipment, 'models/le_equipment.pkl')
joblib.dump(le_level, 'models/le_level.pkl')
joblib.dump(le_goal, 'models/le_goal.pkl')
print("Processed data and encoders saved")