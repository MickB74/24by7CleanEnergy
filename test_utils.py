import utils
import pandas as pd
import io
import os
import zipfile

def test_synthetic_data():
    print("Testing synthetic data generation...")
    # Test with new portfolio structure
    portfolio = [
        {'type': 'Office', 'annual_mwh': 1000},
        {'type': 'Warehouse', 'annual_mwh': 500}
    ]
    df = utils.generate_synthetic_8760_data(year=2023, building_portfolio=portfolio)
    
    assert len(df) == 8760
    assert 'Solar' in df.columns
    assert 'Wind' in df.columns
    assert 'Load' in df.columns
    
    # Check if individual building columns exist
    # The names are dynamic, so check for partial matches
    cols = df.columns.tolist()
    assert any('Load_Office' in c for c in cols)
    assert any('Load_Warehouse' in c for c in cols)
    
    print("Synthetic data test passed.")
    return df

def test_calculations(df):
    print("Testing calculations...")
    results, df_res = utils.calculate_portfolio_metrics(df, solar_capacity=100, wind_capacity=100, load_scaling=1.0)
    
    assert 'total_annual_load' in results
    assert 'cfe_percent' in results
    assert 0 <= results['cfe_percent'] <= 100
    assert 'Hourly_CFE_Ratio' in df_res.columns
    print("Calculations test passed.")
    print(f"CFE %: {results['cfe_percent']}")

def test_export(df):
    print("Testing export...")
    results, _ = utils.calculate_portfolio_metrics(df, 100, 100, 1.0)
    zip_bytes = utils.create_zip_export(results, df, "Test Portfolio", "ERCOT")
    assert len(zip_bytes) > 0
    print("Export test passed.")

def test_file_upload():
    print("Testing file upload...")
    # Create dummy CSV
    csv_content = "timestamp,Solar,Wind,Load\n2024-01-01 00:00:00,0,10,50\n2024-01-01 01:00:00,0,12,48"
    dummy_file = io.StringIO(csv_content)
    dummy_file.name = "test.csv"
    
    df = utils.process_uploaded_file(dummy_file)
    assert df is not None
    assert len(df) == 2
    assert 'Solar' in df.columns
    print("File upload test passed.")

def test_zip_upload():
    print("Testing zip upload...")
    # Create dummy CSV
    csv_content = "timestamp,Solar,Wind,Load\n"
    # Add 8765 rows to test truncation as well
    for i in range(8765):
        csv_content += f"2023-01-01 00:00:00,{i},{i},{i}\n"
        
    # Create Zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("data.csv", csv_content)
    
    # Mock uploaded file object
    zip_buffer.seek(0)
    # Streamlit UploadedFile has a .name attribute
    class MockUploadedFile(io.BytesIO):
        def __init__(self, buffer, name):
            super().__init__(buffer.getvalue())
            self.name = name
            
    uploaded_zip = MockUploadedFile(zip_buffer, "test.zip")
    
    df = utils.process_uploaded_file(uploaded_zip)
    
    assert df is not None, "Should return a DataFrame from zip"
    assert len(df) == 8760, f"Should be truncated to 8760 rows, got {len(df)}"
    assert 'Solar' in df.columns
    print("Zip upload and 8760 truncation test passed.")

def test_year_default():
    print("Testing default year 2023...")
    df = utils.generate_synthetic_8760_data(year=2023)
    assert df['timestamp'].dt.year.iloc[0] == 2023
    assert len(df) == 8760
    print("Default year test passed.")

if __name__ == "__main__":
    try:
        df = test_synthetic_data()
        test_calculations(df)
        test_export(df)
        test_file_upload()
        test_zip_upload()
        test_year_default()
        print("\nALL TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
