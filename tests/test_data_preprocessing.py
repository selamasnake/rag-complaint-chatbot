import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pytest
import pandas as pd
from io import StringIO
from data_preprocessing import DataLoader, ComplaintFilter, NarrativeCleaner, ComplaintEDA

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def raw_data_csv():
    """Simulates the raw CFPB CSV structure."""
    data = """Complaint ID,Product,Consumer complaint narrative
1,Credit card,I am writing to file a complaint regarding my card.
2,Savings account,To whom it may concern: My account is locked.
3,Mortgage,This should be filtered out.
4,Credit card or prepaid card,Another credit card complaint.
5,Checking or savings account,XXXX XXXX money is missing.
6,Personal Loan,Missing narrative here.
"""
    return StringIO(data)

@pytest.fixture
def sample_df(raw_data_csv):
    return pd.read_csv(raw_data_csv)

# --- TESTS for ComplaintFilter ---

def test_product_filtering(sample_df):
    """Ensure only target products remain and are mapped correctly."""
    filterer = ComplaintFilter(sample_df)
    df_filtered = filterer.filter_products()
    
    # Check that 'Mortgage' was removed
    assert "Mortgage" not in df_filtered['Product'].values
    # Check that mapping worked
    assert "Credit Card" in df_filtered['product_category'].values
    assert "Savings Account" in df_filtered['product_category'].values

def test_remove_missing_narratives():
    """Ensure rows with NaN narratives are dropped."""
    df = pd.DataFrame({
        'Product': ['Credit card', 'Credit card'],
        'Consumer complaint narrative': ['Text', None]
    })
    filterer = ComplaintFilter(df)
    df_result = filterer.remove_missing_narratives()
    assert len(df_result) == 1

# --- TESTS for NarrativeCleaner ---

def test_text_cleaning_logic():
    test_text = "I AM WRITING TO FILE A COMPLAINT REGARDING my XXXX account!"
    cleaned = NarrativeCleaner.clean_text(test_text)
    
    assert cleaned == cleaned.lower()
    # Change this to match the actual output
    assert "redacted" in cleaned

# --- TESTS for ComplaintEDA ---

def test_narrative_coverage(sample_df):
    """Verify null vs non-null counting logic."""
    # Manually add a null to the sample
    sample_df.loc[5, 'Consumer complaint narrative'] = None 
    eda = ComplaintEDA(sample_df)
    coverage = eda.narrative_coverage()
    
    assert coverage.loc["With Narrative", "count"] == 5
    assert coverage.loc["Without Narrative", "count"] == 1

# --- TESTS for DataLoader ---

def test_save_and_load(tmp_path, sample_df):
    """Test if DataLoader correctly handles file I/O using a temp directory."""
    d_path = tmp_path / "test_processed.csv"
    loader = DataLoader()
    
    # Test Save
    loader.save_data(str(d_path), sample_df)
    assert os.path.exists(d_path)
    
    # Test Load
    loaded_df = loader.load_data(str(d_path))
    assert len(loaded_df) == len(sample_df)
    assert "Complaint ID" in loaded_df.columns