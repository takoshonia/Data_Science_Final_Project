"""
Interactive Visualization Module using Plotly

This module contains functions for creating interactive visualizations
using Plotly for the Urban Pulse project.

 To run the dashboard, use the command: streamlit run dashboard.py
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


def plot_traffic_distribution_interactive(df: pd.DataFrame,
                                         volume_column: str = 'traffic_volume') -> go.Figure:
    """
    Create interactive histogram showing traffic volume distribution.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    volume_column : str, optional
        Name of traffic volume column
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Traffic Volume Distribution', 'Distribution with KDE'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=df[volume_column],
            nbinsx=50,
            name='Traffic Volume',
            marker_color='steelblue',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add mean and median lines
    mean_val = df[volume_column].mean()
    median_val = df[volume_column].median()
    
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.0f}",
        row=1, col=1
    )
    
    fig.add_vline(
        x=median_val,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: {median_val:.0f}",
        row=1, col=1
    )
    
    # KDE plot (using histogram approximation if scipy not available)
    try:
        from scipy import stats
        x_range = np.linspace(df[volume_column].min(), df[volume_column].max(), 100)
        kde = stats.gaussian_kde(df[volume_column])
        y_kde = kde(x_range)
    except ImportError:
        # Fallback: use histogram density
        hist, bins = np.histogram(df[volume_column], bins=50, density=True)
        x_range = (bins[:-1] + bins[1:]) / 2
        y_kde = hist
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_kde,
            mode='lines',
            name='KDE',
            line=dict(color='darkblue', width=2),
            fill='tozeroy',
            opacity=0.5
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Traffic Volume", row=1, col=1)
    fig.update_xaxes(title_text="Traffic Volume", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=2)
    
    fig.update_layout(
        title_text="Traffic Volume Distribution Analysis",
        height=500,
        showlegend=True
    )
    
    return fig


def plot_traffic_by_weekday_interactive(df: pd.DataFrame,
                                       volume_column: str = 'traffic_volume',
                                       day_column: str = 'day_of_week') -> go.Figure:
    """
    Create interactive boxplot and bar chart showing traffic by day of week.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    volume_column : str, optional
        Name of traffic volume column
    day_column : str, optional
        Name of day of week column
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_plot = df.copy()
    
    if df_plot[day_column].dtype != 'object':
        df_plot['day_name'] = df_plot[day_column].map(dict(enumerate(day_names)))
    else:
        df_plot['day_name'] = df_plot[day_column]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Traffic Volume by Day of Week (Boxplot)', 'Average Traffic by Day'),
        specs=[[{"type": "box"}, {"type": "bar"}]]
    )
    
    # Boxplot
    for day in day_names:
        day_data = df_plot[df_plot['day_name'] == day][volume_column]
        if len(day_data) > 0:
            fig.add_trace(
                go.Box(
                    y=day_data,
                    name=day,
                    boxmean='sd'
                ),
                row=1, col=1
            )
    
    # Bar chart with means
    mean_by_day = df_plot.groupby('day_name')[volume_column].mean().reindex(day_names)
    
    fig.add_trace(
        go.Bar(
            x=mean_by_day.index,
            y=mean_by_day.values,
            name='Average Traffic',
            marker_color='coral',
            text=[f'{v:.0f}' for v in mean_by_day.values],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Day of Week", row=1, col=1)
    fig.update_xaxes(title_text="Day of Week", row=1, col=2)
    fig.update_yaxes(title_text="Traffic Volume", row=1, col=1)
    fig.update_yaxes(title_text="Average Traffic Volume", row=1, col=2)
    
    fig.update_layout(
        title_text="Traffic Patterns by Day of Week",
        height=500,
        showlegend=False
    )
    
    return fig


def plot_time_series_interactive(df: pd.DataFrame,
                                 date_column: str = 'date_time',
                                 volume_column: str = 'traffic_volume') -> go.Figure:
    """
    Create interactive time series plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_column : str, optional
        Name of datetime column
    volume_column : str, optional
        Name of traffic volume column
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    df_ts = df.copy()
    df_ts = df_ts.sort_values(date_column)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Traffic Volume Over Time (Full Series)', 'Traffic with Rolling Average'),
        vertical_spacing=0.1
    )
    
    # Full time series
    fig.add_trace(
        go.Scatter(
            x=df_ts[date_column],
            y=df_ts[volume_column],
            mode='lines',
            name='Traffic Volume',
            line=dict(color='steelblue', width=1),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Rolling average
    if len(df_ts) > 24:
        window = min(24, len(df_ts) // 10)
        df_ts['rolling_mean'] = df_ts[volume_column].rolling(window=window).mean()
        
        fig.add_trace(
            go.Scatter(
                x=df_ts[date_column],
                y=df_ts[volume_column],
                mode='lines',
                name='Raw Data',
                line=dict(color='lightblue', width=1),
                opacity=0.3
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_ts[date_column],
                y=df_ts['rolling_mean'],
                mode='lines',
                name=f'{window}-Hour Rolling Average',
                line=dict(color='darkblue', width=2)
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Traffic Volume", row=1, col=1)
    fig.update_yaxes(title_text="Traffic Volume", row=2, col=1)
    
    fig.update_layout(
        title_text="Traffic Volume Time Series Analysis",
        height=700,
        hovermode='x unified'
    )
    
    return fig


def plot_congestion_by_hour_interactive(df: pd.DataFrame,
                                       hour_column: str = 'hour') -> go.Figure:
    """
    Create interactive bar chart showing congestion by hour.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    hour_column : str, optional
        Name of hour column
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Traffic Volume by Hour', 'Congestion Rate by Hour'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Average traffic by hour
    hourly_avg = df.groupby(hour_column)['traffic_volume'].mean().sort_index()
    
    fig.add_trace(
        go.Bar(
            x=hourly_avg.index,
            y=hourly_avg.values,
            name='Average Traffic',
            marker_color='steelblue',
            text=[f'{v:.0f}' for v in hourly_avg.values],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Congestion rate by hour
    if 'is_congested' in df.columns:
        congestion_rate = df.groupby(hour_column)['is_congested'].mean() * 100
        
        fig.add_trace(
            go.Bar(
                x=congestion_rate.index,
                y=congestion_rate.values,
                name='Congestion Rate',
                marker_color='coral',
                text=[f'{v:.1f}%' for v in congestion_rate.values],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # Add threshold line
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_color="red",
            annotation_text="50% Threshold",
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Hour of Day", row=1, col=1, dtick=1)
    fig.update_xaxes(title_text="Hour of Day", row=1, col=2, dtick=1)
    fig.update_yaxes(title_text="Average Traffic Volume", row=1, col=1)
    fig.update_yaxes(title_text="Congestion Rate (%)", row=1, col=2)
    
    fig.update_layout(
        title_text="Traffic Patterns by Hour of Day",
        height=500,
        showlegend=False
    )
    
    return fig


def plot_weather_impact_interactive(df: pd.DataFrame,
                                   weather_column: str = 'weather_main',
                                   volume_column: str = 'traffic_volume') -> go.Figure:
    """
    Create interactive bar chart showing weather impact.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    weather_column : str, optional
        Name of weather column
    volume_column : str, optional
        Name of traffic volume column
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if weather_column not in df.columns:
        raise ValueError(f"Column '{weather_column}' not found in DataFrame")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Traffic by Weather', 'Number of Observations by Weather'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Average traffic by weather
    weather_avg = df.groupby(weather_column)[volume_column].mean().sort_values(ascending=False)
    
    fig.add_trace(
        go.Bar(
            x=weather_avg.values,
            y=weather_avg.index,
            orientation='h',
            name='Average Traffic',
            marker_color='teal',
            text=[f'{v:.0f}' for v in weather_avg.values],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Count by weather
    weather_count = df[weather_column].value_counts()
    
    fig.add_trace(
        go.Bar(
            x=weather_count.values,
            y=weather_count.index,
            orientation='h',
            name='Observations',
            marker_color='skyblue',
            text=[f'{v:,}' for v in weather_count.values],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Average Traffic Volume", row=1, col=1)
    fig.update_xaxes(title_text="Number of Observations", row=1, col=2)
    fig.update_yaxes(title_text="Weather Condition", row=1, col=1)
    fig.update_yaxes(title_text="Weather Condition", row=1, col=2)
    
    fig.update_layout(
        title_text="Weather Impact on Traffic",
        height=600,
        showlegend=False
    )
    
    return fig


def plot_temperature_vs_traffic_interactive(df: pd.DataFrame,
                                            temp_column: str = 'temp',
                                            volume_column: str = 'traffic_volume') -> go.Figure:
    """
    Create interactive scatter plot showing temperature vs traffic.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    temp_column : str, optional
        Name of temperature column (assumed to be in Kelvin)
    volume_column : str, optional
        Name of traffic volume column
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Filter out invalid temperature values (0 or extremely low)
    # Temperature in Kelvin should be > 200K (approximately -73°C, which is reasonable)
    df_clean = df.copy()
    df_clean = df_clean[(df_clean[temp_column] > 200) & (df_clean[temp_column] < 320)]
    
    if len(df_clean) == 0:
        raise ValueError("No valid temperature data after filtering")
    
    # Convert temperature from Kelvin to Celsius for better interpretation
    df_clean['temp_celsius'] = df_clean[temp_column] - 273.15
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Temperature vs Traffic Volume', 'Traffic by Temperature Range'),
        specs=[[{"type": "scatter"}, {"type": "box"}]]
    )
    
    # Scatter plot with Celsius
    fig.add_trace(
        go.Scatter(
            x=df_clean['temp_celsius'],
            y=df_clean[volume_column],
            mode='markers',
            name='Traffic',
            marker=dict(
                color=df_clean[volume_column],
                colorscale='Viridis',
                size=5,
                opacity=0.6,
                showscale=True,
                colorbar=dict(title="Traffic Volume", x=1.15)
            ),
            hovertemplate='<b>Temperature:</b> %{x:.1f}°C<br>' +
                         '<b>Traffic:</b> %{y:.0f} vehicles<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add trend line (only for valid data range)
    z = np.polyfit(df_clean['temp_celsius'], df_clean[volume_column], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df_clean['temp_celsius'].min(), df_clean['temp_celsius'].max(), 100)
    y_trend = p(x_trend)
    
    # Clip trend line to non-negative traffic volumes
    y_trend = np.maximum(y_trend, 0)
    
    fig.add_trace(
        go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            name=f'Trend: y={z[0]:.2f}x+{z[1]:.0f}',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>Trend Line</b><extra></extra>'
        ),
        row=1, col=1
    )
    
    # Boxplot by temperature bins (using Celsius)
    # Create meaningful temperature bins
    temp_min = df_clean['temp_celsius'].min()
    temp_max = df_clean['temp_celsius'].max()
    
    # Use quantiles for better binning
    bins = [
        df_clean['temp_celsius'].quantile(0),
        df_clean['temp_celsius'].quantile(0.2),
        df_clean['temp_celsius'].quantile(0.4),
        df_clean['temp_celsius'].quantile(0.6),
        df_clean['temp_celsius'].quantile(0.8),
        df_clean['temp_celsius'].quantile(1.0)
    ]
    
    labels = ['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot']
    df_clean['temp_bin'] = pd.cut(df_clean['temp_celsius'], bins=bins, labels=labels, include_lowest=True)
    
    # Add boxplots for each bin
    for temp_bin in labels:
        bin_data = df_clean[df_clean['temp_bin'] == temp_bin][volume_column]
        if len(bin_data) > 0:
            fig.add_trace(
                go.Box(
                    y=bin_data,
                    name=temp_bin,
                    boxmean='sd',
                    hovertemplate=f'<b>{temp_bin}</b><br>' +
                                 '<b>Traffic:</b> %{y:.0f} vehicles<br>' +
                                 '<extra></extra>'
                ),
                row=1, col=2
            )
    
    # Set proper axis limits
    fig.update_xaxes(
        title_text="Temperature (°C)",
        row=1, col=1,
        range=[df_clean['temp_celsius'].min() - 5, df_clean['temp_celsius'].max() + 5]
    )
    fig.update_xaxes(title_text="Temperature Range", row=1, col=2)
    fig.update_yaxes(
        title_text="Traffic Volume",
        row=1, col=1,
        range=[0, df_clean[volume_column].max() * 1.1]  # Start at 0, add 10% padding
    )
    fig.update_yaxes(
        title_text="Traffic Volume",
        row=1, col=2,
        range=[0, df_clean[volume_column].max() * 1.1]
    )
    
    fig.update_layout(
        title_text="Temperature Impact on Traffic",
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


def plot_correlation_heatmap_interactive(df: pd.DataFrame,
                                         numeric_columns: Optional[List[str]] = None) -> go.Figure:
    """
    Create interactive correlation heatmap.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    numeric_columns : list, optional
        List of numeric column names to include
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation
    corr_matrix = df[numeric_columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title_text="Feature Correlation Heatmap",
        height=600,
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    return fig


def plot_rush_hour_comparison_interactive(df: pd.DataFrame,
                                         volume_column: str = 'traffic_volume') -> go.Figure:
    """
    Create interactive violin plot comparing rush hour vs non-rush hour.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    volume_column : str, optional
        Name of traffic volume column
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    df_plot = df.copy()
    df_plot['Rush Hour'] = df_plot['is_rush_hour'].map({0: 'Non-Rush Hour', 1: 'Rush Hour'})
    
    fig = go.Figure()
    
    for rush_type in ['Non-Rush Hour', 'Rush Hour']:
        data = df_plot[df_plot['Rush Hour'] == rush_type][volume_column]
        fig.add_trace(go.Violin(
            y=data,
            name=rush_type,
            box_visible=True,
            meanline_visible=True,
            fillcolor='lightblue' if rush_type == 'Non-Rush Hour' else 'lightcoral',
            line_color='black'
        ))
    
    fig.update_layout(
        title_text="Traffic Volume: Rush Hour vs Non-Rush Hour",
        yaxis_title="Traffic Volume",
        xaxis_title="Time Period",
        height=500,
        showlegend=False
    )
    
    return fig

