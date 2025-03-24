from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


class FeatureImportanceStrategy(ABC):
    """
    Abstract base class for computing feature importance.

    Why is this needed?
    --------------------
    - Different methods exist for feature importance (e.g., RandomForest, SHAP, permutation).
    - This abstraction allows flexibility to switch between different strategies.
    - Ensures each implementation follows a consistent interface.
    """

    @abstractmethod
    def compute_importance(self, X: pd.DataFrame, y: pd.Series):
        """
        Computes feature importance. Must be implemented in subclasses.

        Parameters:
        -----------
        - X : pd.DataFrame : Feature dataset.
        - y : pd.Series : Target variable.

        Returns:
        --------
        - A pandas Series containing feature importance scores.
        """
        raise NotImplementedError("Subclasses must implement compute_importance()")

    def plot_importance(
        self, feature_importances: pd.Series, title: str = "Feature Importance"
    ):
        """
        Plots feature importance as a horizontal bar chart.

        Why is this needed?
        --------------------
        - Helps in understanding which features contribute the most to predictions.
        - Can be used for feature selection to remove less important features.

        Parameters:
        -----------
        - feature_importances : pd.Series : Series containing feature importance scores.
        - title : str : Title of the plot.

        Raises:
        -------
        - ValueError : If the provided Series is empty.
        """
        if feature_importances.empty:
            raise ValueError("Feature importances cannot be empty for plotting.")

        # Create a bar plot of feature importance scores
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=feature_importances.values, y=feature_importances.index, palette="viridis"
        )
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.title(title)
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.show()


class RandomForestFeatureImportance(FeatureImportanceStrategy):
    """
    Computes feature importance using a RandomForest model.

    Why is this needed?
    --------------------
    - RandomForest provides built-in feature importance scores based on Gini impurity.
    - It helps in identifying the most relevant features for prediction.
    - The model is robust to multicollinearity and handles non-linearity well.
    """

    def compute_importance(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains a RandomForest model and extracts feature importance scores.

        Parameters:
        -----------
        - X : pd.DataFrame : Feature dataset.
        - y : pd.Series : Target variable.

        Returns:
        --------
        - A pandas Series containing sorted feature importance scores.

        Raises:
        -------
        - ValueError : If X or y is empty.
        """
        if X.empty or y.empty:
            raise ValueError("Input data cannot be empty.")

        # Train a RandomForest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Extract and sort feature importances
        feature_importances = pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        return feature_importances


# Main Execution Block
if __name__ == "__main__":
    pass
