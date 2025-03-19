import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load processed data
dataset = pd.read_csv('data/processed_data.csv')
X = dataset[['Age', 'Weight', 'Goal', 'Equipment', 'Level']]
y = dataset['Title']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, 'models/workout_model.pkl')
print("Model saved as 'workout_model.pkl'")