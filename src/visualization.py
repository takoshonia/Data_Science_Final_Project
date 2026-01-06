"""
Visualization Module for Urban Pulse Project

This module contains functions for creating various types of visualizations
for exploratory data analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_traffic_distribution(df: pd.DataFrame, 
                             volume_column: str = 'traffic_volume',
                             save_path: Optional[str] = None) -> None:
    """
    Create histogram showing traffic volume distribution.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    volume_column : str, optional
        Name of traffic volume column (default: 'traffic_volume')
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(df[volume_column], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Traffic Volume', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Traffic Volume Distribution (Histogram)', fontsize=14, fontweight='bold')
    axes[0].axvline(df[volume_column].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df[volume_column].mean():.0f}')
    axes[0].axvline(df[volume_column].median(), color='green', linestyle='--', 
                    label=f'Median: {df[volume_column].median():.0f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # KDE Plot
    axes[1].hist(df[volume_column], bins=50, density=True, alpha=0.5, color='steelblue', label='Histogram')
    df[volume_column].plot(kind='kde', ax=axes[1], color='darkblue', linewidth=2, label='KDE')
    axes[1].set_xlabel('Traffic Volume', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Traffic Volume Distribution (KDE)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_traffic_by_weekday(df: pd.DataFrame,
                            volume_column: str = 'traffic_volume',
                            day_column: str = 'day_of_week',
                            save_path: Optional[str] = None) -> None:
    """
    Create boxplot showing traffic volume by day of week.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    volume_column : str, optional
        Name of traffic volume column
    day_column : str, optional
        Name of day of week column
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Boxplot
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    df_box = df.copy()
    if df_box[day_column].dtype != 'object':
        df_box['day_name'] = df_box[day_column].map(dict(enumerate(day_names)))
    else:
        df_box['day_name'] = df_box[day_column]
    
    sns.boxplot(data=df_box, x='day_name', y=volume_column, ax=axes[0], palette='Set2')
    axes[0].set_xlabel('Day of Week', fontsize=12)
    axes[0].set_ylabel('Traffic Volume', fontsize=12)
    axes[0].set_title('Traffic Volume by Day of Week (Boxplot)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Bar chart with means
    mean_by_day = df_box.groupby('day_name')[volume_column].mean()
    axes[1].bar(mean_by_day.index, mean_by_day.values, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Day of Week', fontsize=12)
    axes[1].set_ylabel('Average Traffic Volume', fontsize=12)
    axes[1].set_title('Average Traffic Volume by Day of Week', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_time_series(df: pd.DataFrame,
                    date_column: str = 'date_time',
                    volume_column: str = 'traffic_volume',
                    save_path: Optional[str] = None) -> None:
    """
    Create time series plot showing traffic volume over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_column : str, optional
        Name of datetime column
    volume_column : str, optional
        Name of traffic volume column
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    df_ts = df.copy()
    df_ts = df_ts.sort_values(date_column)
    
    # Full time series
    axes[0].plot(df_ts[date_column], df_ts[volume_column], linewidth=0.5, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Traffic Volume', fontsize=12)
    axes[0].set_title('Traffic Volume Over Time (Full Series)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Rolling average
    if len(df_ts) > 24:  # Only if we have enough data points
        window = min(24, len(df_ts) // 10)  # 24-hour rolling average
        df_ts['rolling_mean'] = df_ts[volume_column].rolling(window=window).mean()
        axes[1].plot(df_ts[date_column], df_ts[volume_column], linewidth=0.3, alpha=0.5, 
                    color='lightblue', label='Raw Data')
        axes[1].plot(df_ts[date_column], df_ts['rolling_mean'], linewidth=2, 
                    color='darkblue', label=f'{window}-Hour Rolling Average')
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Traffic Volume', fontsize=12)
        axes[1].set_title(f'Traffic Volume with {window}-Hour Rolling Average', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame,
                            numeric_columns: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> None:
    """
    Create correlation heatmap for numeric features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    numeric_columns : list, optional
        List of numeric columns to include. If None, auto-selects.
    save_path : str, optional
        Path to save the figure
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target if it's in the list (we'll add it back separately)
        if 'is_congested' in numeric_columns:
            numeric_columns.remove('is_congested')
    
    # Calculate correlation
    corr_data = df[numeric_columns + ['is_congested'] if 'is_congested' in df.columns else numeric_columns].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))  # Mask upper triangle
    
    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap: Traffic Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_temperature_vs_traffic(df: pd.DataFrame,
                               temp_column: str = 'temp',
                               volume_column: str = 'traffic_volume',
                               save_path: Optional[str] = None) -> None:
    """
    Create scatter plot showing relationship between temperature and traffic.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    temp_column : str, optional
        Name of temperature column
    volume_column : str, optional
        Name of traffic volume column
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    scatter = axes[0].scatter(df[temp_column], df[volume_column], alpha=0.5, 
                            c=df[volume_column], cmap='viridis', s=20)
    axes[0].set_xlabel('Temperature', fontsize=12)
    axes[0].set_ylabel('Traffic Volume', fontsize=12)
    axes[0].set_title('Temperature vs Traffic Volume (Scatter Plot)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[0], label='Traffic Volume')
    axes[0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df[temp_column], df[volume_column], 1)
    p = np.poly1d(z)
    axes[0].plot(df[temp_column], p(df[temp_column]), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend: y={z[0]:.2f}x+{z[1]:.0f}')
    axes[0].legend()
    
    # Boxplot by temperature bins
    df_plot = df.copy()
    df_plot['temp_bin'] = pd.cut(df_plot[temp_column], bins=5, labels=['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot'])
    sns.boxplot(data=df_plot, x='temp_bin', y=volume_column, ax=axes[1], palette='coolwarm')
    axes[1].set_xlabel('Temperature Range', fontsize=12)
    axes[1].set_ylabel('Traffic Volume', fontsize=12)
    axes[1].set_title('Traffic Volume by Temperature Range', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_congestion_by_hour(df: pd.DataFrame,
                           hour_column: str = 'hour',
                           save_path: Optional[str] = None) -> None:
    """
    Create bar chart showing congestion levels by hour of day.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    hour_column : str, optional
        Name of hour column
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average traffic by hour
    hourly_avg = df.groupby(hour_column)['traffic_volume'].mean()
    axes[0].bar(hourly_avg.index, hourly_avg.values, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Hour of Day', fontsize=12)
    axes[0].set_ylabel('Average Traffic Volume', fontsize=12)
    axes[0].set_title('Average Traffic Volume by Hour of Day', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(0, 24))
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Congestion rate by hour (if is_congested exists)
    if 'is_congested' in df.columns:
        congestion_rate = df.groupby(hour_column)['is_congested'].mean() * 100
        axes[1].bar(congestion_rate.index, congestion_rate.values, color='coral', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Hour of Day', fontsize=12)
        axes[1].set_ylabel('Congestion Rate (%)', fontsize=12)
        axes[1].set_title('Congestion Rate by Hour of Day', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(0, 24))
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_rush_hour_comparison(df: pd.DataFrame,
                             volume_column: str = 'traffic_volume',
                             save_path: Optional[str] = None) -> None:
    """
    Create violin plot comparing rush hour vs non-rush hour traffic.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    volume_column : str, optional
        Name of traffic volume column
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Violin plot
    df_plot = df.copy()
    df_plot['Rush Hour'] = df_plot['is_rush_hour'].map({0: 'Non-Rush Hour', 1: 'Rush Hour'})
    
    sns.violinplot(data=df_plot, x='Rush Hour', y=volume_column, ax=axes[0], palette='Set2')
    axes[0].set_xlabel('Time Period', fontsize=12)
    axes[0].set_ylabel('Traffic Volume', fontsize=12)
    axes[0].set_title('Traffic Volume: Rush Hour vs Non-Rush Hour (Violin Plot)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Swarm plot (sample if too many points)
    df_sample = df_plot.sample(min(1000, len(df_plot))) if len(df_plot) > 1000 else df_plot
    sns.swarmplot(data=df_sample, x='Rush Hour', y=volume_column, ax=axes[1], 
                 size=2, alpha=0.5, palette='Set2')
    axes[1].set_xlabel('Time Period', fontsize=12)
    axes[1].set_ylabel('Traffic Volume', fontsize=12)
    axes[1].set_title('Traffic Volume: Rush Hour vs Non-Rush Hour (Swarm Plot)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def plot_weather_impact(df: pd.DataFrame,
                       weather_column: str = 'weather_main',
                       volume_column: str = 'traffic_volume',
                       save_path: Optional[str] = None) -> None:
    """
    Create bar chart showing traffic volume by weather condition.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    weather_column : str, optional
        Name of weather column
    volume_column : str, optional
        Name of traffic volume column
    save_path : str, optional
        Path to save the figure
    """
    if weather_column not in df.columns:
        print(f"Warning: Column '{weather_column}' not found in DataFrame")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average traffic by weather
    weather_avg = df.groupby(weather_column)[volume_column].mean().sort_values(ascending=False)
    axes[0].barh(weather_avg.index, weather_avg.values, color='teal', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Average Traffic Volume', fontsize=12)
    axes[0].set_ylabel('Weather Condition', fontsize=12)
    axes[0].set_title('Average Traffic Volume by Weather Condition', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Count by weather
    weather_count = df[weather_column].value_counts()
    axes[1].barh(weather_count.index, weather_count.values, color='skyblue', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Number of Observations', fontsize=12)
    axes[1].set_ylabel('Weather Condition', fontsize=12)
    axes[1].set_title('Number of Observations by Weather Condition', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()


def create_summary_statistics_plot(df: pd.DataFrame,
                                  volume_column: str = 'traffic_volume',
                                  save_path: Optional[str] = None) -> None:
    """
    Create a summary statistics visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    volume_column : str, optional
        Name of traffic volume column
    save_path : str, optional
        Path to save the figure
    """
    stats = df[volume_column].describe()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Statistics bar chart
    stat_names = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    stat_values = [stats['count'], stats['mean'], stats['std'], stats['min'], 
                  stats['25%'], stats['50%'], stats['75%'], stats['max']]
    
    axes[0].bar(stat_names, stat_values, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_title('Summary Statistics for Traffic Volume', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Boxplot with statistics annotations
    bp = axes[1].boxplot(df[volume_column], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[1].set_ylabel('Traffic Volume', fontsize=12)
    axes[1].set_title('Traffic Volume Distribution (Boxplot)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add text annotations
    axes[1].text(1.1, stats['min'], f"Min: {stats['min']:.0f}", fontsize=9, verticalalignment='bottom')
    axes[1].text(1.1, stats['25%'], f"Q1: {stats['25%']:.0f}", fontsize=9, verticalalignment='center')
    axes[1].text(1.1, stats['50%'], f"Median: {stats['50%']:.0f}", fontsize=9, verticalalignment='center')
    axes[1].text(1.1, stats['75%'], f"Q3: {stats['75%']:.0f}", fontsize=9, verticalalignment='center')
    axes[1].text(1.1, stats['max'], f"Max: {stats['max']:.0f}", fontsize=9, verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()

