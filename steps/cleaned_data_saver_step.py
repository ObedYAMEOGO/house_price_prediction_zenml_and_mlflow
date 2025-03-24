import os
import pandas as pd
from zenml import step

CLEANED_DATA_DIR = "data/processed"
CLEANED_DATA_PATH = os.path.join(CLEANED_DATA_DIR, "cleaned_data.csv")

@step
def cleaned_data_saver(cleaned_data: pd.DataFrame) -> None:
    """
    Saves the cleaned dataset to a CSV file.

    Args:
        cleaned_data (pd.DataFrame): The cleaned dataset.
    """
    # Ensure the directory exists
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)

    # Save the DataFrame directly to CSV
    cleaned_data.to_csv(CLEANED_DATA_PATH, index=False)  
    print(f"Cleaned dataset saved to {CLEANED_DATA_PATH}")
