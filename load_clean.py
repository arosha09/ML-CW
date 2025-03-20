import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, on_bad_lines='skip')
        print("Dataset loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def clean_data(data):
    if data.isnull().sum().sum() > 0:
        print("Missing values found. Filling with appropriate values.")
        data = data.fillna(method='ffill')
    data['Sets and Reps'] = data['Sets and Reps'].str.replace(' sec', '')
    data = data.drop_duplicates()
    print("Data cleaning completed.")
    return data

if __name__ == "__main__":
    data = load_data('data/newDataset.csv')
    if data is not None:
        cleaned_data = clean_data(data)
        cleaned_data.to_csv('data/cleaned.csv', index=False)
        print("Cleaned data saved to 'data/cleaned.csv'.")