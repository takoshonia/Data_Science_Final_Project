"""
Urban Pulse - Traffic Analysis Dashboard

Interactive Streamlit dashboard for exploring traffic volume data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.interactive_visualization import (
    plot_traffic_distribution_interactive,
    plot_traffic_by_weekday_interactive,
    plot_time_series_interactive,
    plot_congestion_by_hour_interactive,
    plot_weather_impact_interactive,
    plot_temperature_vs_traffic_interactive,
    plot_correlation_heatmap_interactive,
    plot_rush_hour_comparison_interactive
)

# Page configuration
st.set_page_config(
    page_title="Urban Pulse - Traffic Analysis",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚦 Urban Pulse - Traffic Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load and cache the processed data."""
    try:
        df = pd.read_csv('data/processed/traffic_cleaned.csv', parse_dates=['date_time'])
        return df
    except FileNotFoundError:
        st.error("⚠️ Data file not found! Please run the preprocessing notebook first.")
        return None

df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # Date range filter
    if 'date_time' in df.columns:
        min_date = df['date_time'].min().date()
        max_date = df['date_time'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            df = df[(df['date_time'].dt.date >= date_range[0]) & 
                    (df['date_time'].dt.date <= date_range[1])]
    
    # Weather filter
    if 'weather_main' in df.columns:
        weather_options = ['All'] + sorted(df['weather_main'].unique().tolist())
        selected_weather = st.sidebar.selectbox("Filter by Weather", weather_options)
        
        if selected_weather != 'All':
            df = df[df['weather_main'] == selected_weather]
    
    # Rush hour filter
    if 'is_rush_hour' in df.columns:
        rush_filter = st.sidebar.selectbox(
            "Filter by Time Period",
            ['All', 'Rush Hour Only', 'Non-Rush Hour Only']
        )
        
        if rush_filter == 'Rush Hour Only':
            df = df[df['is_rush_hour'] == 1]
        elif rush_filter == 'Non-Rush Hour Only':
            df = df[df['is_rush_hour'] == 0]
    
    # Key Metrics
    st.header("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_traffic = df['traffic_volume'].mean()
        st.metric("Average Traffic Volume", f"{avg_traffic:,.0f}")
    
    with col2:
        max_traffic = df['traffic_volume'].max()
        st.metric("Peak Traffic Volume", f"{max_traffic:,.0f}")
    
    with col3:
        if 'is_congested' in df.columns:
            congestion_rate = df['is_congested'].mean() * 100
            st.metric("Congestion Rate", f"{congestion_rate:.1f}%")
        else:
            st.metric("Total Records", f"{len(df):,}")
    
    with col4:
        if 'is_weekend' in df.columns:
            weekday_avg = df[df['is_weekend'] == 0]['traffic_volume'].mean()
            weekend_avg = df[df['is_weekend'] == 1]['traffic_volume'].mean()
            diff = ((weekday_avg - weekend_avg) / weekend_avg) * 100
            st.metric("Weekday vs Weekend", f"{diff:.1f}% higher")
        else:
            st.metric("Data Points", f"{len(df):,}")
    
    st.markdown("---")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Distribution", 
        "📅 Time Analysis", 
        "⏰ Hourly Patterns",
        "🌤️ Weather Impact",
        "🌡️ Temperature",
        "🔗 Correlations"
    ])
    
    with tab1:
        st.header("Traffic Volume Distribution")
        fig = plot_traffic_distribution_interactive(df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Rush Hour Comparison")
        fig_rush = plot_rush_hour_comparison_interactive(df)
        st.plotly_chart(fig_rush, use_container_width=True)
    
    with tab2:
        st.header("Time Series Analysis")
        fig = plot_time_series_interactive(df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Traffic by Day of Week")
        fig_weekday = plot_traffic_by_weekday_interactive(df)
        st.plotly_chart(fig_weekday, use_container_width=True)
    
    with tab3:
        st.header("Hourly Traffic Patterns")
        fig = plot_congestion_by_hour_interactive(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.subheader("📌 Insights")
        hourly_avg = df.groupby('hour')['traffic_volume'].mean().sort_values(ascending=False)
        top_hours = hourly_avg.head(3)
        
        col1, col2, col3 = st.columns(3)
        for i, (hour, volume) in enumerate(top_hours.items()):
            with [col1, col2, col3][i]:
                st.metric(f"Peak Hour {hour}:00", f"{volume:,.0f}")
    
    with tab4:
        st.header("Weather Impact on Traffic")
        fig = plot_weather_impact_interactive(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Weather statistics
        if 'weather_main' in df.columns:
            st.subheader("Weather Statistics")
            weather_stats = df.groupby('weather_main')['traffic_volume'].agg(['mean', 'count']).round(0)
            weather_stats.columns = ['Avg Traffic', 'Count']
            weather_stats = weather_stats.sort_values('Avg Traffic', ascending=False)
            st.dataframe(weather_stats, use_container_width=True)
    
    with tab5:
        st.header("Temperature vs Traffic")
        if 'temp' in df.columns:
            fig = plot_temperature_vs_traffic_interactive(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Temperature conversion info
            st.info("💡 Temperature is in Kelvin. To convert: Celsius = Kelvin - 273.15")
        else:
            st.warning("Temperature data not available")
    
    with tab6:
        st.header("Feature Correlations")
        numeric_cols = ['traffic_volume', 'temp', 'rain_1h', 'snow_1h', 'clouds_all',
                       'hour', 'day_of_week', 'is_weekend', 'is_rush_hour']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) > 1:
            fig = plot_correlation_heatmap_interactive(df, numeric_columns=available_cols)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns for correlation analysis")
    
    # Data Explorer
    st.markdown("---")
    st.header("🔍 Data Explorer")
    
    with st.expander("View Raw Data"):
        num_rows = st.slider("Number of rows to display", 10, 1000, 100)
        st.dataframe(df.head(num_rows), use_container_width=True)
    
    with st.expander("Data Summary"):
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Date Range:**", f"{df['date_time'].min()} to {df['date_time'].max()}")
        st.write("**Columns:**", list(df.columns))
        
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing[missing > 0] if missing.sum() > 0 else pd.Series(["No missing values"]))
    
    # Download button
    st.markdown("---")
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_traffic_data.csv",
        mime="text/csv"
    )

else:
    st.error("""
    ## ⚠️ Data Not Found
    
    Please ensure:
    1. The data preprocessing notebook (`02_data_preprocessing.ipynb`) has been run
    2. The processed data file exists at `data/processed/traffic_cleaned.csv`
    
    Run the preprocessing notebook first to generate the required data file.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>Urban Pulse - Traffic Analysis Dashboard</p>
        <p>Built with Streamlit & Plotly</p>
    </div>
""", unsafe_allow_html=True)

