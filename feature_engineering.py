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

    # Add 'Age' and 'Weight' features
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

def prepare_features(data):
    X = data[['Body Part/Muscle_Encoded', 'Equipment_Encoded', 'Age', 'Weight']]
    y = data['Level_Encoded']  # Predict Level instead of Exercise Name
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