import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configure logging for better debugging and tracking
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelBuildingStrategy(ABC):
    """
    Abstract base class for model-building strategies.

    Why is this needed?
    --------------------
    - Enables flexibility to implement multiple ML models.
    - Enforces a common `build_and_train_model` method in all subclasses.
    - Promotes clean, structured, and modular model-building.
    """

    @abstractmethod
    def build_and_train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> RegressorMixin:
        """
        Builds and trains a regression model.

        Parameters:
        -----------
        - X_train : pd.DataFrame : The training feature set.
        - y_train : pd.Series : The target variable for training.

        Returns:
        --------
        - Trained model instance.
        """
        raise NotImplementedError("This method must be implemented in a subclass.")


class LinearRegressionStrategy(ModelBuildingStrategy):
    """
    Implements a Linear Regression model with a preprocessing pipeline.

    Why is this needed?
    --------------------
    - Standardizes input data to improve model performance.
    - Uses a pipeline for better modularity and scalability.
    - Ensures that any data transformation (e.g., scaling) is consistently applied.
    """

    def build_and_train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Pipeline:
        """
        Builds and trains a Linear Regression model with standard scaling.

        Parameters:
        -----------
        - X_train : pd.DataFrame : Feature set for training.
        - y_train : pd.Series : Target variable.

        Returns:
        --------
        - Trained model pipeline.
        """
        # Type checking to prevent common errors
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Linear Regression model with scaling.")

        # Creating a pipeline: First, scale the data; then, apply Linear Regression
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Standardizes input features
                ("model", LinearRegression()),  # Fits a Linear Regression model
            ]
        )

        logging.info("Training Linear Regression model.")
        pipeline.fit(X_train, y_train)  # Fit the pipeline to the training data

        logging.info("Model training completed.")
        return pipeline


class ModelBuilder:
    """
    Handles model-building using different strategies.

    Why is this needed?
    --------------------
    - Allows easy switching between different ML models.
    - Decouples model-building logic from the rest of the application.
    - Promotes code reuse and maintainability.
    """

    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initializes the ModelBuilder with a specified strategy.

        Parameters:
        -----------
        - strategy : ModelBuildingStrategy : The chosen model-building approach.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        """
        Allows switching to a different model-building strategy dynamically.

        Parameters:
        -----------
        - strategy : ModelBuildingStrategy : The new model-building strategy.
        """
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Builds and trains a model using the selected strategy.

        Parameters:
        -----------
        - X_train : pd.DataFrame : Feature set for training.
        - y_train : pd.Series : Target variable.

        Returns:
        --------
        - Trained model instance.
        """
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train)


# Main Execution Block
if __name__ == "__main__":
    pass
