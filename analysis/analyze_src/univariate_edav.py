from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Abstract base class for univariate analysis strategy
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented in subclasses.")


# Concrete strategy for numerical feature analysis
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """Performs univariate analysis for numerical features."""
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame.")
            return

        print(f"\nSummary Statistics for {feature}:")
        print(df[feature].describe())

        # Plot distribution and boxplot
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[feature])
        plt.title(f"Boxplot of {feature}")

        plt.show()


# Concrete strategy for categorical feature analysis
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """Performs univariate analysis for categorical features."""
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame.")
            return

        plt.figure(figsize=(10, 6))
        sns.countplot(
            x=feature, data=df, palette="muted", legend=False
        )  # Fixed warning
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


# Context class for univariate analysis
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """Initialize with a specific analysis strategy."""
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """Set a different analysis strategy."""
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """Execute the selected univariate analysis strategy."""
        self._strategy.analyze(df, feature)


# Main Execution Block
if __name__ == "__main__":
    pass
