import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

def encode_features(data):
    """
    Encode categorical variables and save the encoders.
    Returns the transformed dataset and the encoders.
    """
    # Initialize label encoders
    le_body_part = LabelEncoder()
    le_equipment = LabelEncoder()
    le_level = LabelEncoder()
    le_exercise_name = LabelEncoder()

    # Clean the Level column
    data['Level'] = data['Level'].str.lower().str.strip()

    # Fit label encoders on all possible labels
    le_body_part.fit(data['Body Part/Muscle'])
    le_equipment.fit(data['Equipment'])
    le_level.fit(['easy', 'intermediate', 'hard'])  # Ensure the encoder is trained with these labels
    le_exercise_name.fit(data['Exercise Name'])

    # Encode categorical columns
    data['Body Part/Muscle'] = le_body_part.transform(data['Body Part/Muscle'])
    data['Equipment'] = le_equipment.transform(data['Equipment'])
    data['Level'] = le_level.transform(data['Level'])
    data['Exercise Name'] = le_exercise_name.transform(data['Exercise Name'])

    # Add 'Age' and 'Weight' features
    n_samples = data.shape[0]
    data['Age'] = np.random.randint(18, 60, n_samples)
    data['Weight'] = np.random.uniform(50, 120, n_samples)

    # Save the encoders for later use in the web app
    with open('models/le_body_part.pkl', 'wb') as f:
        pickle.dump(le_body_part, f)
    with open('models/le_equipment.pkl', 'wb') as f:
        pickle.dump(le_equipment, f)
    with open('models/le_level.pkl', 'wb') as f:
        pickle.dump(le_level, f)
    with open('models/le_exercise_name.pkl', 'wb') as f:
        pickle.dump(le_exercise_name, f)

    print("Feature encoding completed.")
    return data, le_body_part, le_equipment, le_level, le_exercise_name

def prepare_features(data):
    """
    Prepare features (X) and target (y) for model training.
    """
    X = data[['Body Part/Muscle', 'Equipment', 'Level', 'Age', 'Weight']]
    y = data['Exercise Name']
    return X, y

if __name__ == "__main__":
    # Test the functions
    data = pd.read_csv('data/cleaned.csv')
    if data is not None:
        encoded_data, _, _, _, _ = encode_features(data)
        X, y = prepare_features(encoded_data)
        print("Features (X):")
        print(X.head())
        print("Target (y):")
        print(y.head())
        
        # Save the modified data to a new CSV file
        encoded_data.to_csv('data/encoded_data.csv', index=False)
        print("Encoded data saved to 'data/encoded_data.csv'")