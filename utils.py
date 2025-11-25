import pandas as pd
import numpy as np
import io
import json
import zipfile
import random

# Typical Hourly Load Profiles (% of Peak Load)
LOAD_PROFILES = {
    'Office':      [10, 8, 7, 6, 7, 10, 25, 45, 70, 85, 95, 100, 100, 95, 90, 90, 85, 80, 65, 45, 30, 20, 15, 12],
    'Data Center': [95, 95, 95, 95, 96, 96, 97, 97, 98, 98, 99, 99, 99, 99, 99, 98, 98, 98, 98, 97, 97, 96, 96, 95],
    'Retail':      [5, 4, 3, 3, 4, 6, 10, 25, 50, 75, 85, 95, 100, 100, 95, 90, 85, 80, 75, 60, 40, 25, 15, 8],
    'Residential': [35, 30, 25, 25, 30, 45, 70, 85, 80, 75, 70, 65, 65, 70, 75, 80, 85, 90, 100, 95, 80, 60, 50, 40],
    'Hospital':    [85, 85, 85, 85, 86, 87, 88, 90, 92, 95, 95, 96, 96, 97, 97, 97, 96, 95, 95, 94, 93, 90, 88, 86],
    'Warehouse':   [15, 12, 10, 10, 12, 20, 35, 50, 65, 75, 85, 90, 90, 85, 80, 80, 70, 60, 50, 40, 30, 25, 20, 18]
}

def generate_load_profile_shape(dates, building_type):
    """
    Generates a normalized load profile shape for a given building type.
    Uses defined 24-hour profiles with added seasonality and noise.
    """
    day_of_year = dates.dayofyear.to_numpy()
    hour_of_day = dates.hour.to_numpy()
    
    # Get base 24-hour profile
    base_profile_24h = np.array(LOAD_PROFILES.get(building_type, LOAD_PROFILES['Office']))
    
    # Map to full year
    # hour_of_day is 0-23, so we can directly index
    profile = base_profile_24h[hour_of_day]
    
    # Add Seasonality (Summer Peak)
    # Peak at day 200 (mid-July), min at day 15 (mid-Jan)
    # Amplitude depends on building type? Let's keep it simple for now.
    # Residential might have higher seasonality (AC/Heating).
    # Data Center might have less.
    
    if building_type == 'Data Center':
        seasonality = 1.0 + 0.05 * np.cos((day_of_year - 200) * 2 * np.pi / 365)
    elif building_type == 'Residential':
        seasonality = 1.0 + 0.4 * np.cos((day_of_year - 200) * 2 * np.pi / 365)
    else:
        seasonality = 1.0 + 0.2 * np.cos((day_of_year - 200) * 2 * np.pi / 365)
        
    # Apply seasonality
    profile = profile * seasonality
    
    # Add Random Noise
    noise = np.random.normal(0, 2, size=len(dates)) # +/- 2% noise
    profile = profile + noise
    
    return np.maximum(profile, 0)

REGIONAL_PARAMS = {
    "National Average": {
        "solar_seasonality": 0.4, "solar_cloud": 0.5,
        "wind_seasonality": 0.2, "wind_daily_amp": 0.3, "wind_peak_hour": 4, "wind_base": 30
    },
    "ERCOT": {
        "solar_seasonality": 0.5, "solar_cloud": 0.3, # Sunny, hot summers
        "wind_seasonality": 0.3, "wind_daily_amp": 0.5, "wind_peak_hour": 2, "wind_base": 35 # Strong night wind
    },
    "CAISO": {
        "solar_seasonality": 0.6, "solar_cloud": 0.2, # Very sunny
        "wind_seasonality": 0.2, "wind_daily_amp": 0.4, "wind_peak_hour": 18, "wind_base": 25 # Evening wind (sea breeze)
    },
    "PJM": {
        "solar_seasonality": 0.5, "solar_cloud": 0.6, # Cloudier
        "wind_seasonality": 0.4, "wind_daily_amp": 0.2, "wind_peak_hour": 14, "wind_base": 28 # Winter peak
    },
    "NYISO": {
        "solar_seasonality": 0.5, "solar_cloud": 0.6,
        "wind_seasonality": 0.4, "wind_daily_amp": 0.2, "wind_peak_hour": 14, "wind_base": 28
    },
    "ISO-NE": {
        "solar_seasonality": 0.5, "solar_cloud": 0.6,
        "wind_seasonality": 0.4, "wind_daily_amp": 0.2, "wind_peak_hour": 14, "wind_base": 28
    },
    "MISO": {
        "solar_seasonality": 0.45, "solar_cloud": 0.5,
        "wind_seasonality": 0.3, "wind_daily_amp": 0.4, "wind_peak_hour": 3, "wind_base": 38 # Strong night wind
    },
    "SPP": {
        "solar_seasonality": 0.45, "solar_cloud": 0.4,
        "wind_seasonality": 0.3, "wind_daily_amp": 0.4, "wind_peak_hour": 3, "wind_base": 40 # Very strong wind
    }
}

def generate_synthetic_8760_data(year=2023, building_portfolio=None, region="National Average"):
    """
    Generates synthetic 8760 hourly data for Solar, Wind, and Load.
    building_portfolio: List of dicts [{'type': 'Office', 'annual_mwh': 1000}, ...]
    region: String, one of the keys in REGIONAL_PARAMS
    Returns a DataFrame with datetime index and columns: 'Solar', 'Wind', 'Load' (Total), plus individual building loads.
    """
    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:00:00', freq='h')
    
    params = REGIONAL_PARAMS.get(region, REGIONAL_PARAMS["National Average"])
    
    # Solar: Peak in summer, zero at night, bell curve during day
    day_of_year = dates.dayofyear.to_numpy()
    hour_of_day = dates.hour.to_numpy()
    
    # Seasonality (peak in summer)
    seasonality = 1 + params["solar_seasonality"] * np.cos((day_of_year - 172) * 2 * np.pi / 365)
    
    # Daily pattern (0 at night, peak at noon)
    daily_pattern = np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12))
    daily_pattern[hour_of_day < 6] = 0
    daily_pattern[hour_of_day > 18] = 0
    
    solar_profile = seasonality * daily_pattern * 100 
    cloud_cover = np.random.beta(2, 5, size=len(dates))
    solar_profile = solar_profile * (1 - cloud_cover * params["solar_cloud"])
    
    # Wind
    wind_seasonality = 1 + params["wind_seasonality"] * np.cos((day_of_year - 15) * 2 * np.pi / 365) 
    wind_daily = 1 + params["wind_daily_amp"] * np.cos((hour_of_day - params["wind_peak_hour"]) * 2 * np.pi / 24) 
    wind_noise = np.random.weibull(2, size=len(dates))
    wind_profile = wind_seasonality * wind_daily * wind_noise * params["wind_base"] 
    wind_profile = np.clip(wind_profile, 0, 100) 
    
    # Load Generation
    df = pd.DataFrame({
        'timestamp': dates,
        'Solar': solar_profile,
        'Wind': wind_profile
    })
    
    total_load = np.zeros(len(dates))
    
    if not building_portfolio:
        # Default fallback
        building_portfolio = [{'type': 'Office', 'annual_mwh': 1000}]
        
    for building in building_portfolio:
        b_type = building.get('type', 'Office')
        target_mwh = building.get('annual_mwh', 1000)
        
        # Generate shape
        raw_profile = generate_load_profile_shape(dates, b_type)
        
        # Scale to target MWh
        current_sum = raw_profile.sum()
        if current_sum > 0:
            scaling_factor = (target_mwh * 1000) / current_sum # Convert MWh to kWh? No, let's stick to MWh.
            # Wait, if target is MWh, and profile is MW (power), then sum(MW * 1h) = MWh.
            # So scaling factor = target_mwh / current_sum
            scaling_factor = target_mwh / current_sum
            final_profile = raw_profile * scaling_factor
        else:
            final_profile = raw_profile
            
        col_name = f"Load_{b_type}_{random.randint(100,999)}" # Unique name in case of duplicates
        # Actually, let's just use type and index if needed, or just append.
        # But user might have multiple "Office" buildings.
        # Let's just call it Load_{Type}. If duplicate, pandas handles it or we should be careful.
        # Simple approach: Load_{Type}
        
        # Check if column exists
        base_name = f"Load_{b_type}"
        count = 1
        while base_name in df.columns:
            count += 1
            base_name = f"Load_{b_type}_{count}"
            
        df[base_name] = final_profile
        total_load += final_profile
        
    df['Load'] = total_load
    
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
    
    # Rename base columns to avoid confusion
    rename_map = {
        'Solar': 'Solar Capacity Factor',
        'Wind': 'Wind Capacity Factor',
        'Load': 'Load Profile'
    }
    df = df.rename(columns=rename_map)
    
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
