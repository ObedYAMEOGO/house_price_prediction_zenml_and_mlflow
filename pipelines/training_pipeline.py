from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitting_step import data_splitter_step
from steps.feature_importance_step import RandomForestFeatureImportance
from steps.feature_engineering_step import feature_engineering_step
from steps.missing_values_handler_step import missing_values_handler_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outliers_handling_step import outlier_handling_step
from zenml import Model, pipeline, step


@pipeline(
    model=Model(name="housing_price_predictor"),
)
def ml_pipeline():
    """
    End-to-End Machine Learning Pipeline for House Price Prediction.

    Steps:
    1. Data Ingestion: Load raw dataset.
    2. Missing Values Handling: Fill missing values.
    3. Feature Engineering: Transform selected features.
    4. Outlier Handling: Detect and remove outliers.
    5. Data Splitting: Train-test split.
    6. Model Training: Train a predictive model.
    7. Model Evaluation: Assess model performance.
    8. Feature Importance: Analyze feature contributions.

    Returns:
        Trained model.
    """

    # Step 1: Load raw dataset
    raw_dataset = data_ingestion_step(
        file_path="/mnt/c/Users/debo/Desktop/my_zenml_projects/house_price_prediction_zenml_and_mlflow/data/archive.zip"
    )

    # Step 2: Handle missing values
    nan_filled_dataset = missing_values_handler_step(raw_dataset)

    # Step 3: Perform feature engineering
    engineered_dataset = feature_engineering_step(
        nan_filled_dataset, strategy="log", features=["Gr Liv Area", "SalePrice"]
    )

    # Step 4: Handle outliers
    cleaned_data = outlier_handling_step(engineered_dataset, column_name="SalePrice")

    # Step 5: Split data into training and testing sets
    X_train, X_test, y_train, y_test = data_splitter_step(
        cleaned_data, target_column="SalePrice"
    )

    # Step 6: Train the model
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Step 7: Evaluate model performance
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
