import os
import pandas as pd
import zipfile
from abc import ABC, abstractmethod


# Abstract Base Class for Data Ingestion
# --------------------------------------
# Defines a blueprint for different data ingestion strategies.
# Any subclass must implement the `ingest` method.
class DataIngestor(ABC):
    """Abstract base class for data ingestion."""

    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Extracts and loads data from a file into a Pandas DataFrame."""
        raise NotImplementedError("This method must be implemented in a subclass.")


# Concrete Ingestor: ZIP File Handler
# -----------------------------------
# When to Use:
# - When your dataset is stored inside a ZIP archive.
# - Assumes there is at least one CSV file inside the ZIP.
class ZipFileDataIngestor(DataIngestor):
    """Ingestor for extracting and reading CSV files from a ZIP archive."""

    def ingest(self, file_path: str) -> pd.DataFrame:
        """Extracts a CSV file from a ZIP archive and loads it into a Pandas DataFrame."""

        # Ensure the provided file is a ZIP file
        if not file_path.lower().endswith(".zip"):
            raise ValueError("The provided file isn't a valid ZIP file.")

        extract_to = "extracted_data"  # Directory where files will be extracted

        # Create the extraction directory if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)

        # Extract ZIP archive
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

            # Find all CSV files in the extracted content
            csv_files = [f for f in zip_ref.namelist() if f.lower().endswith(".csv")]

            if not csv_files:
                raise FileNotFoundError("No CSV file found in the ZIP archive.")

            # Use the first CSV file found
            csv_path = os.path.join(extract_to, csv_files[0])

            # Load CSV data into a Pandas DataFrame
            df = pd.read_csv(csv_path)
            print(f"Successfully loaded '{csv_files[0]}' from ZIP archive.")

            return df


# Factory Class for Data Ingestion
# --------------------------------
# - Dynamically selects the correct ingestion strategy based on file type.
# - Can be extended to support more file types in the future.
class DataIngestorFactory:
    """Factory class to select the appropriate data ingestion strategy."""

    @staticmethod
    def get_data_ingestor(file_path: str) -> DataIngestor:
        """Returns an appropriate DataIngestor based on the file extension."""
        if file_path.lower().endswith(".zip"):
            return ZipFileDataIngestor()
        else:
            raise ValueError(
                "Unsupported file type. Only ZIP files are currently supported."
            )


# Main Execution Block
if __name__ == "__main__":
    pass
