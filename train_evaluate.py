import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from load_clean import load_data, clean_data
from feature_engineering import encode_features, prepare_features

def extract_sets_reps_features(data):
    """Extract numerical features from 'Sets and Reps' column."""
    data = data.copy()
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
    data['Number_of_Sets'] = data['Number_of_Sets'].fillna(data['Number_of_Sets'].median())
    data['Min_Reps'] = data['Min_Reps'].fillna(data['Min_Reps'].median())
    data['Max_Reps'] = data['Max_Reps'].fillna(data['Max_Reps'].median())
    
    return data

def add_exercise_complexity(data):
    """Add a feature for exercise complexity based on muscle groups and equipment."""
    data = data.copy()
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

def train_models(X, y):
    """
    Train multiple machine learning models and return the best one.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Debug: Print shapes before SMOTE
    print(f"X_train shape before SMOTE: {X_train.shape}")
    print(f"X_test shape before SMOTE: {X_test.shape}")

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Debug: Print shapes after SMOTE
    print(f"X_train shape after SMOTE: {X_train.shape}")
    print(f"X_test shape after SMOTE: {X_test.shape}")

    # Initialize models
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

    # Print results
    print("Model Evaluation Results:")
    for name, metrics in results.items():
        print(f"{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Plot evaluation metrics
    metrics_df = pd.DataFrame(results).T
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.savefig('model_performance.png')
    plt.close()

    # Choose the best model (based on F1-score)
    best_model_name = max(results, key=lambda x: results[x]['F1'])
    best_model = models[best_model_name]
    print(f"Best model: {best_model_name} with F1-score: {results[best_model_name]['F1']:.4f}")

    # Retrain the best model on the full dataset
    best_model.fit(X, y)

    # Save the best model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    return best_model, results

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('data/cleaned.csv')
    if data is not None:
        cleaned_data = clean_data(data)
        print("Columns after cleaning:", cleaned_data.columns.tolist())
        
        # Add new features
        cleaned_data = extract_sets_reps_features(cleaned_data)
        print("Columns after extract_sets_reps_features:", cleaned_data.columns.tolist())
        
        cleaned_data = add_exercise_complexity(cleaned_data)
        print("Columns after add_exercise_complexity:", cleaned_data.columns.tolist())
        
        # Encode features
        encoded_data, _, _, _ = encode_features(cleaned_data)  # Fixed: Unpack 4 values instead of 5
        print("Columns after encoding:", encoded_data.columns.tolist())
        
        # Prepare features
        X, y = prepare_features(encoded_data)
        print(f"Shape of X before splitting: {X.shape}")
        
        # Train and evaluate models
        best_model, results = train_models(X, y)