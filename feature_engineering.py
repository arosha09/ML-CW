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

    # Encode categorical columns
    data['Body Part/Muscle'] = le_body_part.fit_transform(data['Body Part/Muscle'])
    data['Equipment'] = le_equipment.fit_transform(data['Equipment'])
    data['Level'] = le_level.fit_transform(data['Level'])

    # Save the encoders for later use in the web app
    with open('models/le_body_part.pkl', 'wb') as f:
        pickle.dump(le_body_part, f)
    with open('models/le_equipment.pkl', 'wb') as f:
        pickle.dump(le_equipment, f)
    with open('models/le_level.pkl', 'wb') as f:
        pickle.dump(le_level, f)

    print("Feature encoding completed.")
    return data, le_body_part, le_equipment, le_level

def prepare_features(data):
    """
    Prepare features (X) and target (y) for model training.
    """
    X = data[['Body Part/Muscle', 'Equipment', 'Level']]
    y = data['Exercise Name']
    return X, y

if __name__ == "__main__":
    # Test the functions
    data = pd.read_csv('data/cleaned.csv')
    if data is not None:
        encoded_data, _, _, _ = encode_features(data)
        X, y = prepare_features(encoded_data)
        print("Features (X):")
        print(X.head())
        print("Target (y):")
        print(y.head())