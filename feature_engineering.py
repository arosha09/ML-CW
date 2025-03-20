import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

def encode_features(data):
    le_body_part = LabelEncoder()
    le_equipment = LabelEncoder()
    le_level = LabelEncoder()

    # Clean and encode 'Level' (target variable)
    data['Level'] = data['Level'].str.lower().str.strip()
    le_level.fit(['easy', 'intermediate', 'hard'])
    data['Level_Encoded'] = le_level.transform(data['Level'])

    # Encode features
    le_body_part.fit(data['Body Part/Muscle'])
    le_equipment.fit(data['Equipment'])
    data['Body Part/Muscle_Encoded'] = le_body_part.transform(data['Body Part/Muscle'])
    data['Equipment_Encoded'] = le_equipment.transform(data['Equipment'])

    # Extract numerical features from 'Sets and Reps'
    data = extract_sets_reps_features(data)
    
    # Add exercise complexity features
    data = add_exercise_complexity(data)

    # Add 'Age' and 'Weight' features with reasonable distributions
    n_samples = data.shape[0]
    data['Age'] = np.random.randint(18, 60, n_samples)
    data['Weight'] = np.random.uniform(50, 120, n_samples)

    # Save encoders
    with open('models/le_body_part.pkl', 'wb') as f:
        pickle.dump(le_body_part, f)
    with open('models/le_equipment.pkl', 'wb') as f:
        pickle.dump(le_equipment, f)
    with open('models/le_level.pkl', 'wb') as f:
        pickle.dump(le_level, f)

    print("Feature encoding completed.")
    return data, le_body_part, le_equipment, le_level

def extract_sets_reps_features(data):
    """Extract numerical features from 'Sets and Reps' column."""
    data['Number_of_Sets'] = data['Sets and Reps'].str.extract(r'(\d+)\s*sets').astype(float)
    data['Min_Reps'] = data['Sets and Reps'].str.extract(r'(\d+)-').astype(float)
    data['Max_Reps'] = data['Sets and Reps'].str.extract(r'-(\d+)').astype(float)
    
    # Handle cases where reps are a single number (e.g., "3 sets of 10")
    single_reps = data['Sets and Reps'].str.extract(r'sets of (\d+)').astype(float)
    data['Min_Reps'] = data['Min_Reps'].fillna(single_reps[0])
    data['Max_Reps'] = data['Max_Reps'].fillna(single_reps[0])
    
    # Handle cases with seconds (e.g., "3 sets of 30-60")
    data['Min_Reps'] = data['Min_Reps'].fillna(data['Sets and Reps'].str.extract(r'sets of (\d+)').astype(float)[0])
    data['Max_Reps'] = data['Max_Reps'].fillna(data['Min_Reps'])
    
    # Fill any remaining NaNs with median values
    data['Number_of_Sets'] = data['Number_of_Sets'].fillna(3)  # Default to 3 sets
    data['Min_Reps'] = data['Min_Reps'].fillna(10)  # Default to 10 reps
    data['Max_Reps'] = data['Max_Reps'].fillna(12)  # Default to 12 reps
    
    return data

def add_exercise_complexity(data):
    """Add a feature for exercise complexity based on muscle groups and equipment."""
    # Count number of muscle groups (e.g., "Chest, Shoulders" -> 2)
    data['Muscle_Group_Count'] = data['Body Part/Muscle'].str.split(',').apply(len)
    
    # Flag full-body exercises
    data['Is_Full_Body'] = data['Body Part/Muscle'].str.contains('Full Body', case=False, na=False).astype(int)
    
    # Rank equipment by complexity (simplified)
    equipment_complexity = {
        'Bodyweight': 1, 'Dumbbells': 2, 'Barbell': 3, 'Kettlebell': 2, 'Pull-Up Bar': 2,
        'Cable Machine': 2, 'Machine': 2, 'Box': 1, 'Bench': 1, 'Parallel Bars': 2,
        'Medicine Ball': 2, 'Jump Rope': 1, 'Sled': 3, 'Tire': 3, 'Weight Plate': 2,
        'Lat Pulldown Machine': 2, 'Leg Press Machine': 2, 'Hack Squat Machine': 2,
        'Step/Bench': 1, 'EZ Bar': 2, 'Trap Bar': 3
    }
    data['Equipment_Complexity'] = data['Equipment'].map(equipment_complexity).fillna(1)
    
    return data

def prepare_features(data):
    """Prepare features for model training."""
    # Ensure all required columns are present
    required_columns = [
        'Body Part/Muscle_Encoded', 'Equipment_Encoded', 'Number_of_Sets', 
        'Min_Reps', 'Max_Reps', 'Muscle_Group_Count', 'Is_Full_Body', 
        'Equipment_Complexity'
    ]
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        # Add missing columns with default values
        for col in missing_columns:
            if col == 'Number_of_Sets':
                data[col] = 3
            elif col in ['Min_Reps', 'Max_Reps']:
                data[col] = 10
            else:
                data[col] = 0  # Default value for other columns
    
    X = data[required_columns]
    y = data['Level_Encoded']
    return X, y

if __name__ == "__main__":
    data = pd.read_csv('data/cleaned.csv')
    if data is not None:
        encoded_data, _, _, _ = encode_features(data)
        X, y = prepare_features(encoded_data)
        print("Features (X):")
        print(X.head())
        print("Target (y):")
        print(y.head())
        encoded_data.to_csv('data/encoded_data.csv', index=False)
        print("Encoded data saved to 'data/encoded_data.csv'")