import pandas as pd
import pytest

def test_schema_validation():
    # Simulate loading data
    data = {'isFraud': [0, 1], 'TransactionAmt': [100.0, 50.0]}
    df = pd.DataFrame(data)
    
    # Check if required columns exist
    assert 'isFraud' in df.columns
    assert 'TransactionAmt' in df.columns
    
def test_missing_values():
    data = {'isFraud': [0, 1], 'TransactionAmt': [100.0, None]}
    df = pd.DataFrame(data)
    # Check if we can detect missing values
    assert df['TransactionAmt'].isnull().any() == True
