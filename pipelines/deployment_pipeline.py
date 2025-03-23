import os

# Importing necessary pipeline steps and modules
from pipelines.training_pipeline import ml_pipeline  # Training pipeline
from steps.dynamic_data_importer import dynamic_importer  # Data importer for inference
from steps.prediction_service_loader import (
    prediction_service_loader,
)  # MLflow service loader
from steps.predictor import predictor  # Prediction step
from zenml import pipeline  # ZenML pipeline decorator
from zenml.integrations.mlflow.steps import (
    mlflow_model_deployer_step,
)  # MLflow deployer step

# Define the path to the requirements file (useful for dependency management)
requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


# =============================
#  Continuous Deployment Pipeline
# =============================
@pipeline
def continuous_deployment_pipeline():
    """
    This pipeline automates the training and deployment of an ML model.
    - Runs the training pipeline to train a new model.
    - Deploys the newly trained model using MLflow.
    """

    # Train the model using the existing training pipeline
    trained_model = ml_pipeline()  # Returns the trained model

    # Deploy the trained model to an MLflow service
    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=trained_model)

    # Explanation:
    # - workers=3 enables parallel inference (useful for handling multiple requests at once)
    # - deploy_decision=True ensures the trained model gets deployed automatically
    # - MLflow model deployment allows easy versioning and model tracking


# =============================
#  Inference Pipeline (Batch Prediction)
# =============================
@pipeline(enable_cache=False)
def inference_pipeline():
    """
    This pipeline performs batch inference on new data.
    - Loads new batch data for prediction.
    - Retrieves the latest deployed model.
    - Runs the model on the new data to generate predictions.
    """

    # Step 1: Load new data dynamically (e.g., from an API, database, or file)
    batch_data = dynamic_importer()

    # Step 2: Load the latest deployed model
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",  # Fetch from deployment pipeline
        step_name="mlflow_model_deployer_step",  # Load the deployed model
    )

    # Step 3: Run inference on the batch data using the deployed model
    predictor(service=model_deployment_service, input_data=batch_data)

    # Explanation:
    # - dynamic_importer() retrieves new test data.
    # - prediction_service_loader() ensures the most recent model is used.
    # - predictor() sends data to the deployed model for predictions.
