import pandas as pd

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path, on_bad_lines='skip')
        print("Dataset loaded successfully.")
        print("Initial columns:", data.columns.tolist())
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def clean_data(data):
    """Clean the dataset by handling missing values and ensuring required columns."""
    # Create a copy to avoid modifying the original data
    data = data.copy()
    
    # Ensure required columns are present
    required_columns = ['Exercise Name', 'Body Part/Muscle', 'Equipment', 'Sets and Reps', 'Level']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")
    
    # Handle missing values with appropriate defaults
    if data.isnull().sum().sum() > 0:
        print("Missing values found. Filling with appropriate values.")
        data['Body Part/Muscle'] = data['Body Part/Muscle'].fillna('Unknown')
        data['Equipment'] = data['Equipment'].fillna('Bodyweight')
        data['Sets and Reps'] = data['Sets and Reps'].fillna('3 sets of 10-15')
        data['Level'] = data['Level'].fillna('Intermediate')
        data['Exercise Name'] = data['Exercise Name'].fillna('Unknown Exercise')
    
    # Clean 'Sets and Reps' column
    data['Sets and Reps'] = data['Sets and Reps'].str.replace(' sec', '')
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    print("Data cleaning completed.")
    return data

if __name__ == "__main__":
    data = load_data('data/newDataset.csv')
    if data is not None:
        cleaned_data = clean_data(data)
        cleaned_data.to_csv('data/cleaned.csv', index=False)
        print("Cleaned data saved to 'data/cleaned.csv'.")