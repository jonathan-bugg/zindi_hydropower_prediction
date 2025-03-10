# Hydropower Prediction Data Pipeline

This repository contains a flexible data pipeline for preparing training data to predict hydropower generation (kwh) using time series data and climate variables.

## Overview

The data pipeline is designed to:

1. Aggregate the main dataframe (`df`) at daily x source level
2. Aggregate the climate dataframe (`df_climate`) at daily level
3. Merge the aggregated datasets to create a comprehensive set of features
4. Generate additional time-based features such as lag features and rolling window features
5. Prepare the final training data for model building

## Key Files

- `data_pipeline.py`: The core pipeline implementation with functions for aggregation and feature engineering
- `data_pipeline_demo.ipynb`: A Jupyter notebook demonstrating how to use the pipeline

## Usage

### Basic Usage

```python
from data_pipeline import prepare_training_data

# Define aggregation configurations
df_agg_config = {
    'kwh': ['sum', 'mean', 'std', 'min', 'max'],
    'current': ['mean', 'max'],
    'power_factor': 'mean',
    'v_red': ['mean', 'min', 'max']
}

climate_agg_config = {
    'temperature': ['min', 'max', 'mean'],
    'humidity': 'mean',
    'precipitation': 'sum',
    'wind_speed': ['mean', 'max']
}

# Prepare training data
training_data, feature_cols, target_col = prepare_training_data(
    df,                    # Main dataframe
    df_climate,            # Climate dataframe
    df_agg_config,         # Main df aggregation config
    climate_agg_config,    # Climate df aggregation config
    target_col='kwh_sum'   # Target column for prediction
)
```

### Adding Lag Features

```python
from data_pipeline import create_lag_features

# Create lag features
lag_cols = ['kwh_mean', 'temperature_mean']
lags = [1, 7]  # 1-day and 7-day lags

training_data_with_lags = create_lag_features(
    training_data, 
    lag_cols, 
    lags
)
```

### Adding Rolling Window Features

```python
from data_pipeline import create_rolling_features

# Create rolling window features
rolling_cols = ['kwh_sum', 'temperature_mean']
windows = [3, 7]  # 3-day and 7-day windows
agg_funcs = ['mean', 'std']  # Mean and standard deviation

training_data_with_rolling = create_rolling_features(
    training_data, 
    rolling_cols, 
    windows, 
    agg_funcs
)
```

## Customization

The data pipeline is highly customizable:

- Aggregation functions can be specified for each column separately
- Target column can be changed to any aggregated column
- Lag periods and rolling window sizes can be adjusted
- Additional feature engineering steps can be added

## Data Requirements

- The main dataframe (`df`) must contain:
  - A date column (default: 'date')
  - A source column (default: 'source')
  - The kwh column and other features

- The climate dataframe (`df_climate`) must contain:
  - A date column (default: 'date')
  - Climate-related features

## Extending the Pipeline

The modular design makes it easy to extend the pipeline:

1. Add new feature engineering functions to `data_pipeline.py`
2. Import and use them in your workflow

## Example

For a complete example, see the `data_pipeline_demo.ipynb` notebook, which demonstrates the entire workflow from loading data to preparing the final training dataset.