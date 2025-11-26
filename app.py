import streamlit as st
import pandas as pd
import utils
import time
import random
import altair as alt

# Page Config
st.set_page_config(
    page_title="24/7 Clean Energy Portfolio Analyzer",
    page_icon="⚡",
    layout="wide"
)

# Initialize Session State
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Sidebar - Portfolio Definition
if st.sidebar.button("Start New Analysis", key="sidebar_reset", type="primary"):
    st.session_state.analysis_complete = False
    st.rerun()

st.sidebar.header("Portfolio Configuration")

portfolio_name = st.sidebar.text_input("Portfolio Name", "My Green Portfolio")
region = st.sidebar.selectbox("Region", ["ERCOT", "PJM", "CAISO", "MISO", "SPP", "NYISO", "ISO-NE"])

st.sidebar.subheader("Generation Assets (MW)")
solar_capacity = st.sidebar.number_input("Solar Capacity (MW)", min_value=0.0, value=50.0, step=1.0)
wind_capacity = st.sidebar.number_input("Wind Capacity (MW)", min_value=0.0, value=50.0, step=1.0)



# Main Content
st.title("24/7 Clean Energy Portfolio Analyzer")

# Random Quote
quotes = [
    "The future of energy is not just about generation, but about matching supply with demand, every hour of every day.",
    "Clean energy is the only energy that will count in the long run.",
    "A 24/7 carbon-free grid is the ultimate destination.",
    "Every hour of green energy counts towards a sustainable future.",
    "Decarbonization happens one hour at a time."
]
quote = random.choice(quotes)

# Start Screen / Intro
if not st.session_state.analysis_complete:
    st.markdown(f"""
    ### Welcome to the 24/7 Clean Energy Analyzer
    
    > "{quote}"
    
    **What this tool does:**
    *   **Builds** renewable energy portfolios.
    *   **Analyzes** hourly performance (8,760 hours).
    *   **Compares** generation vs. load to calculate CFE scores.
    *   **Exports** detailed datasets and summaries.
    
    **Get Started:**
    Configure your portfolio below, then click **Run Analysis**.
    """)

    st.subheader("Load Configuration")
    # Default portfolio
    if 'saved_portfolio' in st.session_state and st.session_state.saved_portfolio is not None:
        default_portfolio = st.session_state.saved_portfolio
    else:
        default_portfolio = pd.DataFrame([
            {"Building Type": "Office", "Annual Consumption (MWh)": 0},
            {"Building Type": "Data Center", "Annual Consumption (MWh)": 0},
            {"Building Type": "Retail", "Annual Consumption (MWh)": 0},
            {"Building Type": "Residential", "Annual Consumption (MWh)": 0},
            {"Building Type": "Hospital", "Annual Consumption (MWh)": 0},
            {"Building Type": "Warehouse", "Annual Consumption (MWh)": 0}
        ])

    c1, c2 = st.columns([1, 1])
    
    with c1:
        edited_portfolio = st.data_editor(
            default_portfolio,
            num_rows="dynamic",
            column_config={
                "Building Type": st.column_config.SelectboxColumn(
                    "Building Type",
                    options=["Office", "Warehouse", "Data Center", "Retail", "Residential", "Hospital"],
                    required=True
                ),
                "Annual Consumption (MWh)": st.column_config.NumberColumn(
                    "Annual Consumption (MWh)",
                    min_value=0,
                    step=10,
                    format="%d"
                )
            },
            hide_index=True,
            use_container_width=True
        )

    uploaded_file = st.file_uploader("Upload Custom Data (CSV/XLSX/ZIP)", type=['csv', 'xlsx', 'zip'])

    if st.button("Run Analysis", type="primary"):
        # Save current configuration to session state
        st.session_state.saved_portfolio = edited_portfolio
        
        with st.spinner("Simulating 8,760-hour year..."):
            # Data Loading / Generation
            df_upload = None
            if uploaded_file:
                df_upload = utils.process_uploaded_file(uploaded_file)
            
            if df_upload is not None:
                df = df_upload
                st.success("File uploaded successfully!")
            else:
                if uploaded_file:
                    st.error("Error processing file. Please ensure it has valid columns (Solar, Wind, Load). Using synthetic data instead.")
                
                # Prepare portfolio for generation (Convert kWh to MWh)
                # edited_portfolio has "Annual Consumption (kWh)"
                portfolio_list = []
                for _, row in edited_portfolio.iterrows():
                    val = row['Annual Consumption (MWh)']
                    annual_mwh = val if pd.notnull(val) else 0.0
                    portfolio_list.append({
                        'type': row['Building Type'],
                        'annual_mwh': annual_mwh
                    })
                
                # Generate Data
                # Use 'region' from sidebar (defined at top of file)
                df = utils.generate_synthetic_8760_data(year=2023, building_portfolio=portfolio_list, region=region)
            
            # Calculation
            # load_scaling is now implicitly handled by the MWh inputs, so we pass 1.0 or remove it.
            # But calculate_portfolio_metrics still expects it. Let's pass 1.0.
            results, df_result = utils.calculate_portfolio_metrics(df, solar_capacity, wind_capacity, load_scaling=1.0, region=region)
            
            # Store in session state
            st.session_state.portfolio_data = {
                "results": results,
                "df": df_result,
                "name": portfolio_name,
                "region": region
            }
            st.session_state.analysis_complete = True
            st.rerun()

# Analysis Results
else:
    data = st.session_state.portfolio_data
    results = data['results']
    df = data['df']
    
    # Standard First Summary
    st.markdown("### Portfolio Summary (Standard Format)")
    
    summary_data = {
        "Metric": [
            "Portfolio name",
            "Region",
            "Total annual load (MWh)",
            "Total renewable generation (MWh)",
            "Annual renewable percent (%)",
            "Carbon Free Energy - CFE (%)",
            "Loss of green hour (%)",
            "Overgeneration (MWh)",
            "Grid consumption (MWh)",
            "Grid Emissions (Metric Tons CO2e)",
            "Avoided Emissions (Metric Tons CO2e)"
        ],
        "Value": [
            str(data['name']),
            str(data['region']),
            f"{results['total_annual_load']:,.2f}",
            f"{results['total_renewable_gen']:,.2f}",
            f"{results['annual_re_percent']:.2f}%",
            f"{results['cfe_percent']:.2f}%",
            f"{results['loss_of_green_hour_percent']:.2f}%",
            f"{results['overgeneration']:,.2f}",
            f"{results['grid_consumption']:,.2f}",
            f"{results['grid_emissions_mt']:,.2f}",
            f"{results['avoided_emissions_mt']:,.2f}"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)
    
    st.markdown(f"**Notes:** Synthetic data used for simulation.")
    
    st.divider()
    
    # Detailed Analysis
    st.subheader("Detailed Analysis")
    
    # Monthly Analysis
    df['Month'] = df['timestamp'].dt.month_name()
    
    # Aggregate Energy
    monthly_avg = df.groupby('Month')[['Load_Actual', 'Total_Renewable_Gen', 'Solar_Gen', 'Wind_Gen']].sum().reset_index()
    
    # Aggregate CFE % (Average of hourly ratios)
    monthly_cfe = df.groupby('Month')['Hourly_CFE_Ratio'].mean().reset_index()
    # Keep as ratio for formatting
    
    # Merge
    monthly_data = pd.merge(monthly_avg, monthly_cfe[['Month', 'Hourly_CFE_Ratio']], on='Month')
    
    # Sort months correctly
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    monthly_data['Month'] = pd.Categorical(monthly_data['Month'], categories=months, ordered=True)
    monthly_data = monthly_data.sort_values('Month')
    
    st.markdown("#### Monthly Energy Mix & CFE %")
    
    # Base chart
    base = alt.Chart(monthly_data).encode(
        x=alt.X('Month', sort=months, title='Month')
    )
    
    # Bars (Energy)
    # Melt for stacked bars
    # Bars (Generation Only)
    monthly_gen = monthly_data.melt('Month', value_vars=['Solar_Gen', 'Wind_Gen'], var_name='Type', value_name='Energy')
    # Rename for cleaner legend
    monthly_gen['Type'] = monthly_gen['Type'].replace({'Solar_Gen': 'Solar', 'Wind_Gen': 'Wind'})
    
    # Define unified color domain and range
    domain = ['Solar', 'Wind', 'Total Load', 'CFE %']
    range_ = ['#1f77b4', '#aec7e8', 'magenta', 'red'] # Solar Blue, Wind Light Blue, Load Magenta, CFE Red
    
    bars = alt.Chart(monthly_gen).mark_bar().encode(
        x=alt.X('Month', sort=months),
        y=alt.Y('Energy', title='Total Energy (MWh)', axis=alt.Axis(format=',.0f')),
        color=alt.Color('Type:N', scale=alt.Scale(domain=domain, range=range_), title='Legend'),
        tooltip=[
            alt.Tooltip('Month', title='Month'),
            alt.Tooltip('Type', title='Type'),
            alt.Tooltip('Energy', title='Energy (MWh)', format=',.0f')
        ]
    )
    
    # Line (Load)
    # We use transform_calculate to create a 'Type' field for the legend
    load_line = base.mark_line(strokeDash=[5, 5]).transform_calculate(
        Type="'Total Load'"
    ).encode(
        y=alt.Y('Load_Actual', title='Total Energy (MWh)'),
        color=alt.Color('Type:N', scale=alt.Scale(domain=domain, range=range_), title='Legend'),
        tooltip=[
            alt.Tooltip('Month', title='Month'),
            alt.Tooltip('Load_Actual', title='Load (MWh)', format=',.0f')
        ]
    )
    
    # Line (CFE %)
    cfe_line = base.mark_line(point=True).transform_calculate(
        Type="'CFE %'"
    ).encode(
        y=alt.Y('Hourly_CFE_Ratio', title='CFE %', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Type:N', scale=alt.Scale(domain=domain, range=range_), title='Legend'),
        tooltip=[
            alt.Tooltip('Month', title='Month'),
            alt.Tooltip('Hourly_CFE_Ratio', title='CFE %', format='.1%')
        ]
    )
    
    # Combine
    # Layer bars and load_line (share axis)
    energy_chart = alt.layer(bars, load_line).resolve_scale(y='shared').properties(
        height=300,
        title="Monthly Energy Mix (MWh)"
    )
    
    # CFE Chart (Lower Panel)
    cfe_chart = cfe_line.properties(
        height=100,
        title="CFE %"
    )
    
    # Vertical Concatenation
    chart_combined = alt.vconcat(energy_chart, cfe_chart).resolve_scale(
        x='shared'
    ).configure_view(
        stroke=None
    )
    
    st.altair_chart(chart_combined, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Annual Hourly Profile")
        # Prepare data for Altair - Use full year
        # Create base chart
        base_annual = alt.Chart(df).encode(x=alt.X('timestamp', title='Time'))
        
        # Define colors for legend
        domain = ['Generation', 'Load']
        range_ = ['#1f77b4', 'magenta']
        
        # Generation Line (Blue)
        gen_chart = base_annual.mark_line(strokeWidth=1, opacity=0.7).transform_calculate(
            Profile="'Generation'"
        ).encode(
            y=alt.Y('Total_Renewable_Gen', title='Power (MW)'),
            color=alt.Color('Profile:N', scale=alt.Scale(domain=domain, range=range_), title='Profile'),
            tooltip=[alt.Tooltip('timestamp', title='Time'), alt.Tooltip('Total_Renewable_Gen', title='Generation (MW)', format=',.0f')]
        )
        
        # Load Line (Magenta, on top)
        load_chart = base_annual.mark_line(strokeWidth=1.5).transform_calculate(
            Profile="'Load'"
        ).encode(
            y=alt.Y('Load_Actual', title='Power (MW)'),
            color=alt.Color('Profile:N', scale=alt.Scale(domain=domain, range=range_), title='Profile'),
            tooltip=[alt.Tooltip('timestamp', title='Time'), alt.Tooltip('Load_Actual', title='Load (MW)', format=',.0f')]
        )
        
        chart_annual = alt.layer(gen_chart, load_chart).resolve_scale(y='shared').interactive()
        
        st.altair_chart(chart_annual, use_container_width=True)
        st.caption("Load vs Generation (Full Year)")
        
    with col2:
        st.markdown("#### Duration Curve (Gen & Load)")
        # Sort generation and load descending independently
        gen_sorted = df['Total_Renewable_Gen'].sort_values(ascending=False).reset_index(drop=True)
        load_sorted = df['Load_Actual'].sort_values(ascending=False).reset_index(drop=True)
        
        duration_df = pd.DataFrame({
            'Hour': gen_sorted.index, 
            'Generation': gen_sorted.values,
            'Load': load_sorted.values
        })
        
        duration_long = duration_df.melt('Hour', var_name='Type', value_name='Power')
        
        chart_duration = alt.Chart(duration_long).mark_line().encode(
            x=alt.X('Hour', title='Hours', axis=alt.Axis(format=',.0f'), scale=alt.Scale(domain=[0, 8760], nice=False)),
            y=alt.Y('Power', title='Power (MW)', axis=alt.Axis(format=',.0f')),
            color=alt.Color('Type', title='Profile'),
            tooltip=[
                alt.Tooltip('Hour', title='Hour', format=',.0f'),
                alt.Tooltip('Type', title='Profile'),
                alt.Tooltip('Power', title='Power (MW)', format=',.0f')
            ]
        ).interactive()
        
        st.altair_chart(chart_duration, use_container_width=True)
        st.caption("Sorted Duration Curves (High to Low)")
    
    # Exports
    st.subheader("Downloads")
    zip_data = utils.create_zip_export(results, df, data['name'], data['region'])
    
    st.download_button(
        label="Download Portfolio Package (ZIP)",
        data=zip_data,
        file_name=f"{data['name']}_analysis.zip",
        mime="application/zip"
    )
    
    if st.button("Start New Analysis"):
        st.session_state.analysis_complete = False
        st.rerun()

    st.divider()
    with st.expander("Understand and Adjust Assumptions"):
        st.markdown("""
        **Data Generation Logic:**
        *   **Capacity Factors:** The Solar and Wind inputs represent the *Hourly Capacity Factor* (0-100% availability). They are scaled by your selected Capacity (MW) to calculate actual Generation.
        *   **Solar:** Based on typical meteorological year patterns with seasonal and daily variations.
        *   **Wind:** Modeled with Weibull distribution and seasonal/daily patterns.
        *   **Load:** Synthetic profiles based on building type (Office, Residential, Data Center) with realistic daily/seasonal shapes.
        
        **Calculations:**
        *   **CFE %:** Average of hourly CFE ratios (capped at 100%).
        *   **Loss of Green Hour:** Percentage of hours where renewable generation < load.
        *   **Overgeneration:** Renewable energy produced in excess of load.
        
        **Adjustments:**
        *   Use the sidebar to scale the load or change capacities.
        *   Upload your own 8760 CSV for precise analysis.
        """)
        
        st.markdown("**Data Preview (First 5 Rows):**")
        st.dataframe(df.head())

    with st.expander("Raw 8760 Data"):
        st.dataframe(df)
