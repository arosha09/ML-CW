import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import pickle
from imblearn.over_sampling import SMOTE
from load_clean import load_data, clean_data
from feature_engineering import encode_features, prepare_features

def extract_sets_reps_features(data):
    """Extract numerical features from 'Sets and Reps' column."""
    data['Number_of_Sets'] = data['Sets and Reps'].str.extract(r'(\d+)\s*sets').astype(float)
    data['Min_Reps'] = data['Sets and Reps'].str.extract(r'(\d+)-').astype(float)
    data['Max_Reps'] = data['Sets and Reps'].str.extract(r'-(\d+)').astype(float)
    
    single_reps = data['Sets and Reps'].str.extract(r'sets of (\d+)').astype(float)
    data['Min_Reps'] = data['Min_Reps'].fillna(single_reps[0])
    data['Max_Reps'] = data['Max_Reps'].fillna(single_reps[0])
    
    data['Min_Reps'] = data['Min_Reps'].fillna(data['Sets and Reps'].str.extract(r'sets of (\d+)').astype(float)[0])
    data['Max_Reps'] = data['Max_Reps'].fillna(data['Min_Reps'])
    
    data['Number_of_Sets'] = data['Number_of_Sets'].fillna(data['Number_of_Sets'].median())
    data['Min_Reps'] = data['Min_Reps'].fillna(data['Min_Reps'].median())
    data['Max_Reps'] = data['Max_Reps'].fillna(data['Max_Reps'].median())
    
    return data

def add_exercise_complexity(data):
    """Add a feature for exercise complexity based on muscle groups and equipment."""
    data['Muscle_Group_Count'] = data['Body Part/Muscle'].str.split(',').apply(len)
    data['Is_Full_Body'] = data['Body Part/Muscle'].str.contains('Full Body', case=False, na=False).astype(int)
    
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
    """Update feature preparation with new features."""
    print("Columns in data before preparing features:", data.columns.tolist())
    
    required_columns = [
        'Body Part/Muscle_Encoded', 'Equipment_Encoded', 'Number_of_Sets', 
        'Min_Reps', 'Max_Reps', 'Muscle_Group_Count', 'Is_Full_Body', 
        'Equipment_Complexity'
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    X = data[required_columns]
    y = data['Level_Encoded']
    return X, y

def objective(trial, X_train, y_train, X_test, y_test):
    """Objective function for Optuna hyperparameter tuning using xgb.train."""
    param = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=param['n_estimators'],
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Create an XGBClassifier instance to use the trained booster for prediction
    model = xgb.XGBClassifier(**param, use_label_encoder=False)
    model._Booster = bst  # Assign the trained booster to the classifier
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='weighted')

def train_models(X, y):
    # Split data
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
    
    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50)
    
    # Train the best model using xgb.train
    best_params = study.best_params
    best_params['objective'] = 'multi:softmax'
    best_params['num_class'] = 3
    best_params['eval_metric'] = 'mlogloss'
    best_params['use_label_encoder'] = False
    
    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Train with early stopping
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(
        best_params,
        dtrain,
        num_boost_round=best_params['n_estimators'],
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Create an XGBClassifier instance to use the trained booster
    best_model = xgb.XGBClassifier(**best_params)
    best_model._Booster = bst  
    print(f"X_test shape before prediction: {X_test.shape}")
    
    y_pred = best_model.predict(X_test)
    results = {
        'XGBoost': {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
    }
    
    print("Model Evaluation Results:")
    for name, metrics in results.items():
        print(f"{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Plot results
    metrics_df = pd.DataFrame(results).T
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.savefig('model_performance.png')
    plt.close()
    
    # Save the best model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    return best_model, results

if __name__ == "__main__":
    data = load_data('data/cleaned.csv')
    if data is not None:
        cleaned_data = clean_data(data)
        print("Columns after cleaning:", cleaned_data.columns.tolist())
        cleaned_data = extract_sets_reps_features(cleaned_data)
        print("Columns after extract_sets_reps_features:", cleaned_data.columns.tolist())
        cleaned_data = add_exercise_complexity(cleaned_data)
        print("Columns after add_exercise_complexity:", cleaned_data.columns.tolist())
        encoded_data, _, _, _ = encode_features(cleaned_data)
        print("Columns after encoding:", encoded_data.columns.tolist())
        X, y = prepare_features(encoded_data)
        print(f"Shape of X before splitting: {X.shape}")
        best_model, results = train_models(X, y)