import pandas as pd
import numpy as np
import io
import json
import zipfile
import random

def generate_synthetic_8760_data(year=2023, building_type='Office'):
    """
    Generates synthetic 8760 hourly data for Solar, Wind, and Load.
    Returns a DataFrame with datetime index and columns: 'Solar', 'Wind', 'Load'.
    """
    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:00:00', freq='h')
    
    # Solar: Peak in summer, zero at night, bell curve during day
    # Simple model: Seasonality * Daily Pattern * Random Noise
    day_of_year = dates.dayofyear.to_numpy()
    hour_of_day = dates.hour.to_numpy()
    
    # Seasonality (peak in summer)
    seasonality = 1 + 0.4 * np.cos((day_of_year - 172) * 2 * np.pi / 365)
    
    # Daily pattern (0 at night, peak at noon)
    daily_pattern = np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12))
    daily_pattern[hour_of_day < 6] = 0
    daily_pattern[hour_of_day > 18] = 0
    
    solar_profile = seasonality * daily_pattern * 100 # Scale to 100 MW capacity roughly
    # Add some random cloud cover
    cloud_cover = np.random.beta(2, 5, size=len(dates))
    solar_profile = solar_profile * (1 - cloud_cover * 0.5)
    
    # Wind: Higher in winter/night, more stochastic
    # Simple model: Seasonality * Daily Pattern + Noise
    wind_seasonality = 1 + 0.2 * np.cos((day_of_year - 15) * 2 * np.pi / 365) # Peak in winter
    wind_daily = 1 + 0.3 * np.cos((hour_of_day - 4) * 2 * np.pi / 24) # Peak at night
    
    # Weibull distribution for wind speed to power conversion simulation
    wind_noise = np.random.weibull(2, size=len(dates))
    wind_profile = wind_seasonality * wind_daily * wind_noise * 30 # Scale
    wind_profile = np.clip(wind_profile, 0, 100) # Cap at 100 MW
    
    # Load Generation based on Building Type
    load_base = 50
    
    if building_type == 'Data Center':
        # Flat load, very little variation
        load_profile = np.full(len(dates), load_base) + np.random.normal(0, 1, size=len(dates))
    elif building_type == 'Residential':
        # Morning and Evening peaks
        load_seasonality = 1 + 0.3 * np.cos((day_of_year - 200) * 2 * np.pi / 365) # Summer peak
        daily_p = 1 + 0.5 * np.sin((hour_of_day - 18) * np.pi / 12) + 0.3 * np.sin((hour_of_day - 7) * np.pi / 12)
        load_profile = load_base * load_seasonality * daily_p + np.random.normal(0, 5, size=len(dates))
    else: # Office / Commercial
        # Day peak (9-5), low night
        load_seasonality = 1 + 0.3 * np.cos((day_of_year - 200) * 2 * np.pi / 365) # Summer peak
        daily_p = 1 + 0.6 * np.sin((hour_of_day - 12) * np.pi / 12)
        daily_p[hour_of_day < 7] = 0.3 # Low night load
        daily_p[hour_of_day > 19] = 0.3
        load_profile = load_base * load_seasonality * daily_p + np.random.normal(0, 5, size=len(dates))
    
    load_profile = np.maximum(load_profile, 5) # Min load
    
    df = pd.DataFrame({
        'timestamp': dates,
        'Solar': solar_profile,
        'Wind': wind_profile,
        'Load': load_profile
    })
    
    return df

def calculate_portfolio_metrics(df, solar_capacity, wind_capacity, load_scaling=1.0):
    """
    Calculates portfolio metrics based on inputs.
    df: DataFrame with 'Solar', 'Wind', 'Load' columns (normalized or base profiles)
    solar_capacity: MW
    wind_capacity: MW
    load_scaling: Multiplier for the base load profile
    """
    # Scale profiles
    # Assuming input DF columns are already scaled or represent 1 unit/MW. 
    # For synthetic data above, they are arbitrary. Let's normalize them to 1 MW capacity first if we want to scale.
    # But for simplicity, let's assume the user inputs "Capacity" which scales the profile.
    # To do this correctly with synthetic data, we should treat synthetic data as "1 MW capacity" profile.
    
    # Normalize synthetic data to max 1 for scaling
    if 'Solar' in df.columns and df['Solar'].max() > 0:
        df['Solar_Gen'] = (df['Solar'] / df['Solar'].max()) * solar_capacity
    else:
        df['Solar_Gen'] = 0
        
    if 'Wind' in df.columns and df['Wind'].max() > 0:
        df['Wind_Gen'] = (df['Wind'] / df['Wind'].max()) * wind_capacity
    else:
        df['Wind_Gen'] = 0
        
    if 'Load' in df.columns:
        df['Load_Actual'] = df['Load'] * load_scaling
    else:
        df['Load_Actual'] = 0

    # Total Renewable Generation
    df['Total_Renewable_Gen'] = df['Solar_Gen'] + df['Wind_Gen']
    
    # Metrics
    total_load = df['Load_Actual'].sum()
    total_gen = df['Total_Renewable_Gen'].sum()
    
    # Annual renewable percent
    annual_re_percent = (total_gen / total_load * 100) if total_load > 0 else 0
    
    # Hourly CFE
    # CFE = min(Generation, Load) / Load
    # But wait, "hourly Clean Energy — hourly generation/hourly clean energy, clipped at 1" 
    # The user definition: "hourly Carbon free electricity — hourly generation/hourly clean energy, clipped at 1."
    # Wait, "hourly generation / hourly clean energy" doesn't make sense. It likely means "hourly generation / hourly load".
    # Let's assume: Hourly CFE Ratio = min(Total_Renewable_Gen, Load_Actual) / Load_Actual
    # If Load is 0, handle gracefully.
    
    df['Hourly_CFE_MWh'] = np.minimum(df['Total_Renewable_Gen'], df['Load_Actual'])
    df['Hourly_CFE_Ratio'] = np.where(df['Load_Actual'] > 0, df['Hourly_CFE_MWh'] / df['Load_Actual'], 1.0)
    
    # CFE % = average Carbon Free Energy percentage by hour
    # "one hour can't be greater than 1, so we don't count overgeneration in this."
    cfe_percent = df['Hourly_CFE_Ratio'].mean() * 100
    
    # Loss of green hour (%)
    # "% of hours when renewable generation was less than load"
    loss_of_green_hours_count = (df['Total_Renewable_Gen'] < df['Load_Actual']).sum()
    loss_of_green_hour_percent = (loss_of_green_hours_count / len(df)) * 100
    
    # Overgeneration
    df['Overgeneration_MWh'] = np.maximum(0, df['Total_Renewable_Gen'] - df['Load_Actual'])
    total_overgeneration = df['Overgeneration_MWh'].sum()
    
    # Grid consumption
    df['Grid_Consumption_MWh'] = np.maximum(0, df['Load_Actual'] - df['Total_Renewable_Gen'])
    total_grid_consumption = df['Grid_Consumption_MWh'].sum()
    
    results = {
        "total_annual_load": total_load,
        "total_renewable_gen": total_gen,
        "annual_re_percent": annual_re_percent,
        "cfe_percent": cfe_percent,
        "loss_of_green_hour_percent": loss_of_green_hour_percent,
        "overgeneration": total_overgeneration,
        "grid_consumption": total_grid_consumption
    }
    
    return results, df

def create_zip_export(results, df, portfolio_name, region):
    """
    Creates a zip file containing the JSON summary and the CSV dataset.
    """
    # 1. JSON Summary
    summary_dict = {
        "portfolio_name": portfolio_name,
        "region": region,
        "results": results,
        "metadata": {
            "columns": list(df.columns),
            "generated_at": pd.Timestamp.now().isoformat()
        }
    }
    json_str = json.dumps(summary_dict, indent=4)
    
    # 2. CSV Dataset
    # "csv always needs timestamps for each hour and a new column for hourly Carbon free electricity"
    # We already calculated 'Hourly_CFE_Ratio' in the df.
    # Ensure timestamps are present.
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # 3. Zip File
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(f"{portfolio_name}_summary.json", json_str)
        zip_file.writestr(f"{portfolio_name}_8760_data.csv", csv_buffer.getvalue())
        
    return zip_buffer.getvalue()

def process_uploaded_file(uploaded_file):
    """
    Reads an uploaded CSV or Excel file and standardizes it.
    Expected columns: 'timestamp' (optional), 'Solar', 'Wind', 'Load'.
    If columns are missing, it will try to map common names or fill with zeros/defaults.
    """
    try:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file) as z:
                # Find the CSV file
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    return None
                # Read the first CSV found
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # Standardize column names
        # Simple mapping for demo purposes
        column_map = {
            'Date': 'timestamp', 'Time': 'timestamp', 'datetime': 'timestamp',
            'solar': 'Solar', 'pv': 'Solar', 'Solar Generation': 'Solar',
            'wind': 'Wind', 'Wind Generation': 'Wind',
            'load': 'Load', 'demand': 'Load', 'Consumption': 'Load'
        }
        df = df.rename(columns=column_map)
        
        # Ensure required columns exist
        if 'Solar' not in df.columns:
            df['Solar'] = 0
        if 'Wind' in df.columns:
            df['Wind'] = 0
        if 'Load' not in df.columns:
            # If no load column, maybe it's just generation data? 
            # For now, let's assume we need load. If missing, maybe use synthetic?
            # Let's just init to 0 and let the user know (in a real app)
            df['Load'] = 0
            
        # Ensure 8760 rows
        if len(df) > 8760:
            df = df.iloc[:8760]
            
        # If timestamp is missing, generate it for a non-leap year (2023)
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
        else:
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df
        
    except Exception as e:
        return None
