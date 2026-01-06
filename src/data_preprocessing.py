import os
import re
import pandas as pd

class DataLoader:
    """Load and save CSV datasets."""

    def __init__(self, path: str | None = None):
        self.path = path
        self.df = None

    def load_data(self, path: str | None = None) -> pd.DataFrame:
        file_path = path or self.path
        if not file_path:
            raise ValueError("No file path specified.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.df = pd.read_csv(file_path)
        return self.df

    def save_data(self, output_path: str, df: pd.DataFrame | None = None) -> None:
        df_to_save = df if df is not None else self.df
        if df_to_save is None:
            raise ValueError("No data to save.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_to_save.to_csv(output_path, index=False)


class ComplaintFilter:
    """Filter CFPB complaints to CrediTrust product scope."""

    PRODUCT_MAP = {
        'Credit card': 'Credit Card',
        'Credit card or prepaid card': 'Credit Card',
        'Consumer Loan': 'Personal Loan',
        'Vehicle loan or lease': 'Personal Loan',
        'Savings account': 'Savings Account',
        'Checking or savings account': 'Savings Account',
        'Money transfer, virtual currency, or money service': 'Money Transfer'
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def filter_products(self) -> pd.DataFrame:
        self.df = self.df[self.df['Product'].isin(self.PRODUCT_MAP)]
        self.df['product_category'] = self.df['Product'].map(self.PRODUCT_MAP)
        return self.df

    def remove_missing_narratives(self) -> pd.DataFrame:
        self.df = self.df.dropna(subset=['Consumer complaint narrative'])
        return self.df


class NarrativeCleaner:
    """Clean complaint narratives for semantic embedding."""

    def __init__(self, df: pd.DataFrame, text_col: str):
        self.df = df.copy()
        self.text_col = text_col

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'x{2,}', '[redacted]', text)

        boilerplate_phrases = [
            "i am writing to file a complaint regarding",
            "to whom it may concern",
            "thank you for your time"
        ]

        for phrase in boilerplate_phrases:
            text = text.replace(phrase, "")

        # Keep numbers for financial context
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = " ".join(text.split())
        return text

    def apply(self, output_col: str = "cleaned_narrative") -> pd.DataFrame:
        self.df[output_col] = self.df[self.text_col].apply(self.clean_text)
        return self.df


class ComplaintEDA:
    """EDA helpers for complaint datasets."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def product_distribution(self) -> pd.Series:
        return self.df['product_category'].value_counts()

    def narrative_lengths(self, text_col: str) -> pd.Series:
        return self.df[text_col].apply(lambda x: len(str(x).split()))

    def narrative_coverage(self, text_col="Consumer complaint narrative") -> pd.DataFrame:
        """Return a DataFrame showing complaints with vs without narratives."""
        total = len(self.df)
        with_narrative = self.df[text_col].notna().sum()
        without_narrative = self.df[text_col].isna().sum()
        pct_with = with_narrative / total * 100
        pct_without = without_narrative / total * 100

        coverage_df = pd.DataFrame({
            "count": [with_narrative, without_narrative],
            "percentage": [pct_with, pct_without]
        }, index=["With Narrative", "Without Narrative"])

        return coverage_df
