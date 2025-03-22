import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

def encode_features(data):
    # Initialize label encoders for categorical features
    le_body_part = LabelEncoder()
    le_equipment = LabelEncoder()
    le_level = LabelEncoder()

    # Clean and encode 'Level' feature
    data['Level'] = data['Level'].str.lower().str.strip()
    le_level.fit(['easy', 'intermediate', 'hard'])
    data['Level_Encoded'] = le_level.transform(data['Level'])

    # Encode 'Body Part/Muscle' and 'Equipment' features
    le_body_part.fit(data['Body Part/Muscle'])
    le_equipment.fit(data['Equipment'])
    data['Body Part/Muscle_Encoded'] = le_body_part.transform(data['Body Part/Muscle'])
    data['Equipment_Encoded'] = le_equipment.transform(data['Equipment'])

    # Extract additional features from 'Sets and Reps' column
    data = extract_sets_reps_features(data)
    # Add exercise complexity features
    data = add_exercise_complexity(data)

    # Add random 'Age' and 'Weight' features
    n_samples = data.shape[0]
    data['Age'] = np.random.randint(18, 60, n_samples)
    data['Weight'] = np.random.uniform(50, 120, n_samples)

    # Save label encoders to disk
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
    # Extract number of sets
    data['Number_of_Sets'] = data['Sets and Reps'].str.extract(r'(\d+)\s*sets').astype(float)
    # Extract minimum and maximum reps
    data['Min_Reps'] = data['Sets and Reps'].str.extract(r'(\d+)-').astype(float)
    data['Max_Reps'] = data['Sets and Reps'].str.extract(r'-(\d+)').astype(float)
    # Handle single rep ranges
    single_reps = data['Sets and Reps'].str.extract(r'sets of (\d+)').astype(float)
    data['Min_Reps'] = data['Min_Reps'].fillna(single_reps[0])
    data['Max_Reps'] = data['Max_Reps'].fillna(single_reps[0])
    data['Min_Reps'] = data['Min_Reps'].fillna(data['Sets and Reps'].str.extract(r'sets of (\d+)').astype(float)[0])
    data['Max_Reps'] = data['Max_Reps'].fillna(data['Min_Reps'])
    # Fill missing values with defaults
    data['Number_of_Sets'] = data['Number_of_Sets'].fillna(3) 
    data['Min_Reps'] = data['Min_Reps'].fillna(10)  
    data['Max_Reps'] = data['Max_Reps'].fillna(12)  
    
    return data

def add_exercise_complexity(data):
    """Add a feature for exercise complexity based on muscle groups and equipment."""
    # Count the number of muscle groups involved
    data['Muscle_Group_Count'] = data['Body Part/Muscle'].str.split(',').apply(len)
    # Check if the exercise is a full body exercise
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
    # Define required columns for the model
    required_columns = [
        'Body Part/Muscle_Encoded', 'Equipment_Encoded', 'Number_of_Sets', 
        'Min_Reps', 'Max_Reps', 'Muscle_Group_Count', 'Is_Full_Body', 
        'Equipment_Complexity'
    ]
    
    # Check for missing columns and fill with default values
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        for col in missing_columns:
            if col == 'Number_of_Sets':
                data[col] = 3
            elif col in ['Min_Reps', 'Max_Reps']:
                data[col] = 10
            else:
                data[col] = 0 
    
    # Prepare feature matrix X and target vector y
    X = data[required_columns]
    y = data['Level_Encoded']
    return X, y

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('data/cleaned.csv')
    if data is not None:
        # Encode features and prepare data for model training
        encoded_data, _, _, _ = encode_features(data)
        X, y = prepare_features(encoded_data)
        print("Features (X):")
        print(X.head())
        print("Target (y):")
        print(y.head())
        # Save encoded data to CSV
        encoded_data.to_csv('data/encoded_data.csv', index=False)
        print("Encoded data saved to 'data/encoded_data.csv'")