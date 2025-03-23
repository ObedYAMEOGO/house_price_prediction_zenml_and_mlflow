import logging
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluation import ModelEvaluator, RegressionModelEvaluationStrategy
from zenml import step


@step(enable_cache=False)
def model_evaluator_step(
    trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[dict, float]:
    """
    Evaluates the trained model on test data and returns performance metrics.

    Args:
    -----
    - trained_model (Pipeline): The trained scikit-learn pipeline containing preprocessing and the model.
    - X_test (pd.DataFrame): The feature test dataset.
    - y_test (pd.Series): The actual target values for evaluation.

    Returns:
    --------
    - Tuple[dict, float]: A dictionary containing evaluation metrics and the Mean Squared Error (MSE).
    """

    # Validate input types
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Starting model evaluation...")

    try:
        # Ensure the model contains a preprocessing step before making predictions
        if "preprocessor" not in trained_model.named_steps:
            raise ValueError(
                "The trained model pipeline must include a 'preprocessor' step."
            )

        logging.info("Applying preprocessing to the test dataset...")

        # Convert the transformed output back to a DataFrame
        X_test_processed = pd.DataFrame(
            trained_model.named_steps["preprocessor"].transform(X_test),
            columns=X_test.columns,
        )

        # Initialize model evaluator with regression metrics strategy
        evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())

        logging.info("Evaluating model performance on test data...")
        evaluation_metrics = evaluator.evaluate(
            trained_model.named_steps["model"], X_test_processed, y_test
        )

        # Ensure evaluation results are returned as a dictionary
        if not isinstance(evaluation_metrics, dict):
            raise ValueError("Evaluation metrics must be returned as a dictionary.")

        # Extract Mean Squared Error (MSE) for easy access
        mse = evaluation_metrics.get("Mean Squared Error", None)

        logging.info(f"Model evaluation completed. MSE: {mse:.4f}")
        return evaluation_metrics, mse

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise e
