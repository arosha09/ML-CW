import os
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
    Train multiple machine learning models and generate scatter plots for each algorithm.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Initialize models
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=42, probability=True),  # Enable probability for SVM
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    # Train models and generate scatter plots
    os.makedirs('output', exist_ok=True)  # Ensure the output directory exists

    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict probabilities or classes
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Use probabilities if available
        else:
            y_pred_proba = model.predict(X_test)  # Use predicted classes if probabilities are not available

        # Create a scatter plot
        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(y_test)), y_test, color='blue', label='True Values', alpha=0.6)
        plt.scatter(range(len(y_test)), y_pred_proba, color='red', label='Predicted Values', alpha=0.6)
        
        # Add a prediction line
        plt.plot(range(len(y_test)), y_pred_proba, color='green', label='Prediction Line', linewidth=1.5, alpha=0.8)
        
        plt.title(f'{name} Scatter Plot with Prediction Line')
        plt.xlabel('Sample Index')
        plt.ylabel('Predicted vs True Values')
        plt.legend()

        # Save the scatter plot as a PNG file
        filename = f'output/{name.replace(" ", "_").lower()}_scatter_plot_with_line.png'
        plt.savefig(filename)
        plt.close()

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
        train_models(X, y)