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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from load_clean import load_data, clean_data
from feature_engineering import encode_features, prepare_features

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Decision Tree': (DecisionTreeClassifier(random_state=42), {'max_depth': [None, 10, 20, 30]}),
        'Random Forest': (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100, 200]}),
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
        'SVM': (SVC(random_state=42), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
        'Gradient Boosting': (GradientBoostingClassifier(random_state=42), {'n_estimators': [50, 100, 200]})
    }

    min_class_count = min(np.bincount(y_train))
    n_splits = max(2, min(5, min_class_count))
    if min_class_count < 2:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=2, shuffle=True, random_state=42)
        print("Warning: Stratification not possible due to small class sizes. Using KFold instead.")
    else:
        cv = StratifiedKFold(n_splits=n_splits)

    results = {}
    for name, (model, params) in models.items():
        grid_search = GridSearchCV(model, params, cv=cv, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

    print("Model Evaluation Results:")
    for name, metrics in results.items():
        print(f"{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    metrics_df = pd.DataFrame(results).T
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.savefig('model_performance.png')
    plt.close()

    best_model_name = max(results, key=lambda x: results[x]['F1'])
    best_model = models[best_model_name][0]
    print(f"Best model: {best_model_name} with F1-score: {results[best_model_name]['F1']:.4f}")

    best_model.fit(X, y)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    return best_model, results

if __name__ == "__main__":
    data = load_data('data/cleaned.csv')
    if data is not None:
        cleaned_data = clean_data(data)
        encoded_data, _, _, _ = encode_features(cleaned_data)
        X, y = prepare_features(encoded_data)
        best_model, results = train_models(X, y)