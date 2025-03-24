from abc import ABC, abstractmethod
import pandas as pd


class DataInspectionStrategy(ABC):
    """Abstract base class for data inspection strategies."""

    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Method to be implemented by concrete inspection strategies.

        :param df: Pandas DataFrame to inspect.
        """
        raise NotImplementedError("Subclasses must implement the inspect method.")


class DataTypesInspector(DataInspectionStrategy):
    """Concrete strategy for inspecting data types and basic info."""

    def inspect(self, df: pd.DataFrame):
        """
        Prints DataFrame information, including column types and non-null counts.

        :param df: Pandas DataFrame to inspect.
        """
        df.info()


class DataStatisticalSummaryInspector(DataInspectionStrategy):
    """Concrete strategy for generating summary statistics."""

    def inspect(self, df: pd.DataFrame):
        """
        Prints statistical summaries for numerical and categorical columns.

        :param df: Pandas DataFrame to inspect.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())

        # Inspect categorical features if present
        categorical_columns = df.select_dtypes(include=["object"]).columns
        if categorical_columns.any():
            print("\nSummary Statistics (Categorical Features):")
            print(df.describe(include=["O"]))
        else:
            print("\nNo categorical features to describe.")


class DataInspector:
    """Context class that applies a chosen data inspection strategy."""

    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific strategy.

        :param strategy: Instance of a DataInspectionStrategy subclass.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a different data inspection strategy.

        :param strategy: New instance of a DataInspectionStrategy subclass.
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the currently set inspection strategy.

        :param df: Pandas DataFrame to inspect.
        """
        self._strategy.inspect(df)


# Main Execution Block
if __name__ == "__main__":
    pass
