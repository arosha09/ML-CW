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
from load_clean import load_data, clean_data
from feature_engineering import encode_features, prepare_features

def train_models(X, y):
    """
    Train multiple machine learning models and return the best one.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        encoded_data, _, _, _, _ = encode_features(cleaned_data)
        X, y = prepare_features(encoded_data)

        # Train and evaluate models
        best_model, results = train_models(X, y)