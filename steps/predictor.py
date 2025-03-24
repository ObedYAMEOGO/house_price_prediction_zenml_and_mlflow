import json
import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(service: MLFlowDeploymentService, input_data: str) -> np.ndarray:
    """
    Run an inference request against a deployed MLflow model.

    This function takes input data in JSON format, preprocesses it to match
    the expected model format, and sends it to the deployed MLflow model for prediction.

    Args:
        service (MLFlowDeploymentService): The deployed MLflow service for prediction.
        input_data (str): Input feature data in JSON format.

    Returns:
        np.ndarray: The model's predictions.
    """

    # Ensure the MLflow service is running (if already running, this does nothing)
    service.start(timeout=10)

    # Load the input data from a JSON string into a Python dictionary
    data = json.loads(input_data)

    # Remove unnecessary metadata that might be included in JSON
    data.pop("columns", None)  # Drop column names if present
    data.pop("index", None)  # Drop index values if present

    # Define the expected column names for the model
    expected_columns = [
        "Order",
        "PID",
        "MS SubClass",
        "Lot Frontage",
        "Lot Area",
        "Overall Qual",
        "Overall Cond",
        "Year Built",
        "Year Remod/Add",
        "Mas Vnr Area",
        "BsmtFin SF 1",
        "BsmtFin SF 2",
        "Bsmt Unf SF",
        "Total Bsmt SF",
        "1st Flr SF",
        "2nd Flr SF",
        "Low Qual Fin SF",
        "Gr Liv Area",
        "Bsmt Full Bath",
        "Bsmt Half Bath",
        "Full Bath",
        "Half Bath",
        "Bedroom AbvGr",
        "Kitchen AbvGr",
        "TotRms AbvGrd",
        "Fireplaces",
        "Garage Yr Blt",
        "Garage Cars",
        "Garage Area",
        "Wood Deck SF",
        "Open Porch SF",
        "Enclosed Porch",
        "3Ssn Porch",
        "Screen Porch",
        "Pool Area",
        "Misc Val",
        "Mo Sold",
        "Yr Sold",
    ]

    # Convert the raw JSON data into a pandas DataFrame with expected columns
    df = pd.DataFrame(data["data"], columns=expected_columns)

    # Convert DataFrame to JSON list format that MLflow expects
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))

    # Convert JSON list to a NumPy array (MLflow expects array-like input)
    data_array = np.array(json_list)

    # Run the prediction request to the deployed MLflow model
    prediction = service.predict(data_array)

    # Return the model's predictions
    return prediction
