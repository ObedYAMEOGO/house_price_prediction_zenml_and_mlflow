import logging
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

# Configure logging format for better readability
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataSplittingStrategy(ABC):
    """
    Abstract base class for data splitting strategies.

    Why is this needed?
    --------------------
    - Different ML tasks may require different data splitting approaches.
    - Enforces implementation of the `split_data` method in subclasses.
    - Allows easy switching between various splitting strategies.
    """

    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Splits data into training and testing sets.

        Parameters:
        -----------
        - df : pd.DataFrame : The input dataset.
        - target_column : str : The column name representing the target variable.

        Returns:
        --------
        - X_train, X_test, y_train, y_test : Split datasets.
        """
        raise NotImplementedError("This method must be implemented in a subclass.")


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """
    Implements a basic train-test split strategy.

    Why is this needed?
    --------------------
    - Ensures that the dataset is split into training and testing sets for model evaluation.
    - Allows easy adjustment of test set size and random state for reproducibility.

    Parameters:
    -----------
    - test_size : float : The proportion of the dataset to be used for testing (default = 0.2).
    - random_state : int : Random seed for reproducibility (default = 42).
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Splits the dataset into training and testing sets.

        Parameters:
        -----------
        - df : pd.DataFrame : Input dataset.
        - target_column : str : The column name of the target variable.

        Returns:
        --------
        - X_train, X_test, y_train, y_test : Training and testing data splits.
        """
        logging.info("Performing simple train-test split.")

        # Separate features and target variable
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logging.info(f"Train-test split completed: {self.test_size * 100}% test size.")
        return X_train, X_test, y_train, y_test


class DataSplitter:
    """
    A flexible data splitting handler.

    Why is this needed?
    --------------------
    - Allows switching between different splitting strategies dynamically.
    - Provides a unified interface for dataset splitting.
    """

    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initializes the DataSplitter with a specific splitting strategy.

        Parameters:
        -----------
        - strategy : DataSplittingStrategy : The chosen data splitting strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Allows switching to a different data splitting strategy.

        Parameters:
        -----------
        - strategy : DataSplittingStrategy : The new data splitting strategy.
        """
        logging.info("Switching data splitting strategy.")
        self._strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str):
        """
        Splits the data using the selected strategy.

        Parameters:
        -----------
        - df : pd.DataFrame : The dataset to be split.
        - target_column : str : The column name of the target variable.

        Returns:
        --------
        - X_train, X_test, y_train, y_test : Training and testing datasets.
        """
        logging.info("Splitting data using the selected strategy.")
        return self._strategy.split_data(df, target_column)


# Main Execution Block
if __name__ == "__main__":
    pass
