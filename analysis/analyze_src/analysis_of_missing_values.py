from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class MissingValuesAnalyzer(ABC):
    """
    Abstract base class for analyzing and visualizing missing values in a DataFrame.
    """

    def analyze(self, df: pd.DataFrame):
        """
        Performs missing values identification and visualization.

        :param df: Pandas DataFrame to analyze.
        """
        self.report_missing_values(df)
        self.plot_missing_values(df)

    @abstractmethod
    def report_missing_values(self, df: pd.DataFrame):
        """
        Abstract method to report missing values.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement report_missing_values method"
        )

    @abstractmethod
    def plot_missing_values(self, df: pd.DataFrame):
        """
        Abstract method to visualize missing values.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement plot_missing_values method"
        )


class BasicMissingValuesAnalyzer(MissingValuesAnalyzer):
    """
    Concrete class implementing basic missing values analysis.
    """

    def report_missing_values(self, df: pd.DataFrame):
        """
        Prints a summary of missing values for each column.

        :param df: Pandas DataFrame to analyze.
        """
        print("\nMissing Values Summary:")
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]

        if missing_counts.empty:
            print("No missing values detected.")
        else:
            print(missing_counts)

    def plot_missing_values(self, df: pd.DataFrame):
        """
        Displays a heatmap to visualize missing data patterns.

        :param df: Pandas DataFrame to visualize.
        """
        if df.isnull().sum().sum() == 0:
            print("\nNo missing values detected. Skipping heatmap generation.")
            return

        print("\nGenerating Missing Values Heatmap...")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="plasma")
        plt.title("Missing Values Heatmap")
        plt.show()


# Main Execution Block
if __name__ == "__main__":
    pass
