import pandas as pd

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    try:
        data = pd.read_csv(file_path, on_bad_lines='skip')
        print("Dataset loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def clean_data(data):
    """
    Clean the dataset by handling missing values and inconsistencies.
    """
    # Check for missing values
    if data.isnull().sum().sum() > 0:
        print("Missing values found. Filling with appropriate values.")
        data = data.fillna(method='ffill')  # Forward fill for simplicity

    # Ensure 'Sets and Reps' is in a consistent format (e.g., remove 'sec' for consistency)
    data['Sets and Reps'] = data['Sets and Reps'].str.replace(' sec', '')

    # Remove duplicates if any
    data = data.drop_duplicates()

    print("Data cleaning completed.")
    return data

if __name__ == "__main__":
    # Test the functions
    data = load_data('data/newDataset.csv')
    if data is not None:
        cleaned_data = clean_data(data)
        print(cleaned_data.head())
        # Save the cleaned data to a new CSV file
        cleaned_data.to_csv('data/cleaned.csv', index=False)
        print("Cleaned data saved to 'data/cleaned.csv'.")