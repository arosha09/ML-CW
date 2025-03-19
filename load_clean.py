import pandas as pd

# Load and filter dataset
data = pd.read_csv('data/workoutDataset.csv')
abdominals_data = data[data['BodyPart'] == 'Abdominals'].copy()

# Select relevant columns
abdominals_data = abdominals_data[['Title', 'Type', 'Equipment', 'Level', 'Rating', 'Desc']]

# Handle missing values
abdominals_data['Rating'].fillna(abdominals_data['Rating'].mean(), inplace=True)
abdominals_data.dropna(subset=['Title', 'Type', 'Equipment', 'Level'], inplace=True)

# Save cleaned data
abdominals_data.to_csv('data/cleaned_abdominals.csv', index=False)
print("Cleaned data saved as 'cleaned_abdominals.csv'")