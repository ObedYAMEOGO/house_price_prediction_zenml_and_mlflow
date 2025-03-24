from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class BivariateAnalysis(ABC):
    @abstractmethod
    def plot_relationship(self, df: pd.DataFrame, x_col: str, y_col: str):
        """
        Abstract method to analyze the relationship between two features.
        """
        pass


class ScatterPlotAnalysis(BivariateAnalysis):
    def plot_relationship(self, df: pd.DataFrame, x_col: str, y_col: str):
        """
        Creates a scatter plot for two numerical features.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_col, y=y_col, data=df)
        plt.title(f"{x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()


# In case you have categorical features in your dataset
class BoxPlotAnalysis(BivariateAnalysis):
    def plot_relationship(self, df: pd.DataFrame, x_col: str, y_col: str):
        """
        Creates a box plot for a categorical feature against a numerical feature.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=x_col, y=y_col, data=df)
        plt.title(f"{x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45)
        plt.show()


class BivariateAnalysisExecutor:
    def __init__(self, strategy: BivariateAnalysis):
        """
        Initializes the analysis executor with a specific strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysis):
        """
        Updates the strategy for bivariate analysis.
        """
        self._strategy = strategy

    def execute(self, df: pd.DataFrame, x_col: str, y_col: str):
        """
        Executes the current strategy's analysis method.
        """
        self._strategy.plot_relationship(df, x_col, y_col)

# Main Execution Block
if __name__ == "__main__":
    pass
