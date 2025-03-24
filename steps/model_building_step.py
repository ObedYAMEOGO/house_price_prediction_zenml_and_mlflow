import logging
from typing import Annotated

import pandas as pd # type: ignore
import mlflow # type: ignore
from zenml import step, ArtifactConfig, client, Model
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Retrieve the active experiment tracker from ZenML
experiment_tracker = client.Client().active_stack.experiment_tracker

# Define model metadata for tracking within ZenML
model = Model(
    name="housing_price_predictor",
    version=None,
    license="Apache 2.0",
    description="Housing price prediction model using Linear Regression",
)


@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[
    Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)
]:
    """
    Constructs, trains, and returns a Linear Regression model pipeline.

    The pipeline includes:
    - Preprocessing: Handling missing values and encoding categorical variables.
    - Model Training: Fits a Linear Regression model to the processed data.
    - MLflow Tracking: Logs parameters, metrics, and artifacts.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.

    Returns:
        Pipeline: A trained Scikit-learn pipeline containing preprocessing and model.
    """

    # Input validation to ensure correct data types
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Identify categorical and numerical columns for preprocessing
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Identified categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Identified numerical columns: {numerical_cols.tolist()}")

    # Define preprocessing transformers
    numerical_transformer = SimpleImputer(
        strategy="mean"
    )  # Replaces missing values with column mean
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),  # Imputes missing categorical values
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),  # Converts categories into one-hot encoded features
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Define the complete machine learning pipeline
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
    )

    # Start MLflow experiment tracking (ensures an active run exists)
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        # Enable automatic logging of parameters and metrics
        mlflow.sklearn.autolog()

        logging.info("Training the Linear Regression model...")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed successfully.")

        # Log expected input feature names for consistency in inference
        onehot_encoder = (
            pipeline.named_steps["preprocessor"]
            .transformers_[1][1]
            .named_steps["onehot"]
        )
        onehot_encoder.fit(X_train[categorical_cols])
        expected_columns = numerical_cols.tolist() + list(
            onehot_encoder.get_feature_names_out(categorical_cols)
        )
        logging.info(
            f"Trained model expects the following input columns: {expected_columns}"
        )

    except Exception as e:
        logging.error(f"Error occurred during model training: {e}")
        raise e

    finally:
        # Ensure the MLflow run is closed properly
        mlflow.end_run()

    return pipeline
