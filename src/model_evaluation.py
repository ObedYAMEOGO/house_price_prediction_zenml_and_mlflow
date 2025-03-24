import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging to track execution flow and debugging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelEvaluationStrategy(ABC):
    """
    Abstract base class for model evaluation strategies.

    Why is this needed?
    --------------------
    - Defines a common interface for all evaluation strategies.
    - Allows flexibility to implement different evaluation methods.
    - Ensures consistency in evaluating various ML models.
    """

    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Evaluates a regression model on the test dataset.

        Parameters:
        -----------
        - model : RegressorMixin : Trained regression model.
        - X_test : pd.DataFrame : Feature set for testing.
        - y_test : pd.Series : True target values.

        Returns:
        --------
        - dict : Dictionary containing evaluation metrics.
        """
        raise NotImplementedError("This method must be implemented in a subclass.")


class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    """
    Implements a strategy for evaluating regression models.

    Why is this needed?
    --------------------
    - Standardizes regression model evaluation with key performance metrics.
    - Uses Mean Squared Error (MSE) and R-Squared (R²) for assessment.
    - Ensures easy expansion to include additional evaluation metrics if required.
    """

    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Evaluates the regression model using MSE and R-Squared.

        Parameters:
        -----------
        - model : RegressorMixin : Trained regression model.
        - X_test : pd.DataFrame : Test feature set.
        - y_test : pd.Series : True target values.

        Returns:
        --------
        - dict : Dictionary containing evaluation metrics.
        """
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test must be a pandas DataFrame.")
        if not isinstance(y_test, pd.Series):
            raise TypeError("y_test must be a pandas Series.")

        logging.info("Generating predictions using the trained model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics.")

        # Compute evaluation metrics
        mse = mean_squared_error(y_test, y_pred)  # Measures prediction error
        r2 = r2_score(y_test, y_pred)  # Indicates how well the model explains variance

        metrics = {"Mean Squared Error": mse, "R-Squared": r2}

        logging.info(f"Model Evaluation Completed. Metrics: {metrics}")
        return metrics


class ModelEvaluator:
    """
    Handles model evaluation using different evaluation strategies.

    Why is this needed?
    --------------------
    - Allows easy switching between different evaluation methods.
    - Decouples evaluation logic from the rest of the ML pipeline.
    - Makes evaluation adaptable to different models and requirements.
    """

    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specified strategy.

        Parameters:
        -----------
        - strategy : ModelEvaluationStrategy : The chosen evaluation approach.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Allows dynamic switching to a different model evaluation strategy.

        Parameters:
        -----------
        - strategy : ModelEvaluationStrategy : New model evaluation strategy.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Evaluates the model using the selected strategy.

        Parameters:
        -----------
        - model : RegressorMixin : Trained regression model.
        - X_test : pd.DataFrame : Test feature set.
        - y_test : pd.Series : True target values.

        Returns:
        --------
        - dict : Dictionary containing evaluation metrics.
        """
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)


# Main Execution Block
if __name__ == "__main__":
    pass
