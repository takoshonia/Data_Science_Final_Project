# Data Dictionary - Metro Interstate Traffic Volume Dataset

## Dataset Overview

**Source**: UCI ML Repository / Kaggle  
**Dataset Name**: Metro Interstate Traffic Volume Dataset  
**Description**: Hourly traffic volume data for I-94 Westbound traffic, including weather and temporal features.

## Column Descriptions

### Target Variable

| Column Name | Data Type | Description | Values |
|------------|-----------|-------------|--------|
| `traffic_volume` | int64 | Hourly traffic volume (vehicles per hour) | 0 - 7280 |
| `is_congested` | int64 | Binary classification (derived) | 0 = Not Congested, 1 = Congested |
| `traffic_stress_level` | object | Three-level classification (derived) | Low, Medium, High |

### Temporal Features

| Column Name | Data Type | Description | Values |
|------------|-----------|-------------|--------|
| `date_time` | datetime64 | Timestamp of observation | 2012-10-02 09:00:00 to 2018-09-30 23:00:00 |
| `year` | int64 | Year (derived) | 2012 - 2018 |
| `month` | int64 | Month (derived) | 1 - 12 |
| `day` | int64 | Day of month (derived) | 1 - 31 |
| `hour` | int64 | Hour of day (derived) | 0 - 23 |
| `day_of_week` | int64 | Day of week (derived) | 0 = Monday, 6 = Sunday |
| `is_weekend` | int64 | Weekend flag (derived) | 0 = Weekday, 1 = Weekend |

### Weather Features

| Column Name | Data Type | Description | Values |
|------------|-----------|-------------|--------|
| `temp` | float64 | Temperature in Kelvin | ~240 - 310 K |
| `rain_1h` | float64 | Rainfall in the last hour (mm) | 0.0 - 55.63 |
| `snow_1h` | float64 | Snowfall in the last hour (mm) | 0.0 - 0.51 |
| `clouds_all` | int64 | Cloud cover percentage | 0 - 100 |
| `weather_main` | object | Main weather condition | Clear, Clouds, Rain, Drizzle, Mist, etc. |
| `weather_description` | object | Detailed weather description | Various descriptions |

### Derived Features

| Column Name | Data Type | Description | Values |
|------------|-----------|-------------|--------|
| `is_rush_hour` | int64 | Rush hour flag (derived) | 0 = No, 1 = Yes (7-9 AM, 5-7 PM) |
| `rush_hour_type` | object | Rush hour classification (derived) | normal, morning_rush, evening_rush |

### Other Features

| Column Name | Data Type | Description | Values |
|------------|-----------|-------------|--------|
| `holiday` | object | Holiday indicator | None, Columbus Day, etc. |

## Data Quality Notes

### Missing Values
- **Primary Issue**: `holiday` column has 48,143 missing values (99.87% of rows)
- **Handling Strategy**: Forward-fill (followed by backward-fill for remaining values)
- **Rationale**: 
  - Time series data benefits from forward-fill as it preserves temporal continuity
  - Most days are not holidays, so carrying forward "None" is appropriate
  - Dropping rows would eliminate 99.87% of the dataset, which is unacceptable
  - Mean/median imputation would introduce artificial values that don't reflect temporal patterns
- **Result**: All missing values filled, all 48,204 rows preserved

### Outliers
- **Detection Method**: Interquartile Range (IQR) with factor 1.5
- **Handling Strategy**: Capping (Winsorization) at IQR bounds rather than removal
- **Rationale**:
  - Removing outliers would eliminate valid traffic observations (special events, accidents, weather events)
  - These edge cases are important for the model to learn real-world scenarios
  - Capping preserves all observations while reducing the impact of extreme values
  - Traffic volume outliers capped at [-4417.00, 10543.00] (IQR bounds)
- **Result**: All 48,204 rows preserved; extreme values clipped to reasonable bounds

### Duplicate Rows
- **Count**: 17 duplicate rows detected (0.035% of dataset)
- **Handling Strategy**: Kept in dataset
- **Rationale**:
  - In time series traffic data, identical measurements at different times are valid
  - Example: Same traffic volume at 2 AM on different days is legitimate
  - Low impact (only 0.035% of data)
  - May represent legitimate repeated patterns in traffic behavior
- **Result**: All rows including duplicates retained

### Data Transformations
1. **Datetime Parsing**: `date_time` column parsed to extract temporal features (year, month, day, hour, day_of_week)
2. **Feature Engineering**: Created 10+ derived features including:
   - Rush hour flags (7-9 AM, 5-7 PM based on standard commuter patterns)
   - Traffic stress levels (Low/Medium/High using quantile-based thresholds for balanced distribution)
   - Weekend indicators
3. **Encoding**: Categorical variables encoded for ML models

### Detailed Documentation
For complete documentation of all data cleaning decisions and their rationale, see the **"Data Cleaning Decisions Documentation"** section in `notebooks/02_data_preprocessing.ipynb`.

## Usage Notes

- **Time Series Data**: Data is chronological, so temporal features are important
- **Seasonality**: Consider seasonal patterns (month, day_of_week)
- **Rush Hours**: Strong predictor - 7-9 AM and 5-7 PM show highest traffic
- **Weather Impact**: Adverse weather (rain, snow) typically reduces traffic volume

## References

- Original Dataset: [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)
- Kaggle: [Metro Interstate Traffic Volume](https://www.kaggle.com/datasets)

