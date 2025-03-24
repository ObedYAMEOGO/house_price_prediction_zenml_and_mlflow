from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MultivariateAnalysis(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Perform multivariate analysis by generating a correlation heatmap and a pair plot.
        """
        self.plot_correlation_heatmap(df)
        self.plot_pairplot(df)

    @abstractmethod
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate a heatmap displaying correlations between numerical features.
        """
        raise NotImplementedError("Subclasses must implement plot_correlation_heatmap")

    @abstractmethod
    def plot_pairplot(self, df: pd.DataFrame):
        """
        Generate a pair plot to visualize relationships between features.
        """
        raise NotImplementedError("Subclasses must implement plot_pairplot")


class CorrelationAndPairPlotAnalysis(MultivariateAnalysis):
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generates and displays a correlation heatmap for numerical features.
        """
        plt.figure(figsize=(15, 12))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="magma", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def plot_pairplot(self, df: pd.DataFrame):
        """
        Generates and displays a pair plot to analyze relationships between features.
        """
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Features", y=1.02)
        plt.show()


# Main Execution Block
if __name__ == "__main__":
    pass
