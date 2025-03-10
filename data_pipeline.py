import pandas as pd
import numpy as np
from typing import Dict, List, Union, Callable


def aggregate_df(df: pd.DataFrame, 
                 agg_config: Dict[str, Union[str, List[str]]],
                 date_col: str = 'date',
                 source_col: str = 'source') -> pd.DataFrame:
    """
    Aggregate the main dataframe at daily x source level based on specified aggregation config.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe
    agg_config : dict
        Dictionary with columns as keys and aggregation functions as values
        Example: {'kwh': ['mean', 'sum'], 'current': 'mean'}
    date_col : str
        Column name containing the date
    source_col : str
        Column name containing the source identifier
    
    Returns:
    --------
    pandas DataFrame
        Aggregated dataframe at daily x source level
    """
    # Make sure the date column is datetime type
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
    else:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Group by date and source, then aggregate
    agg_df = df.groupby([date_col, source_col]).agg(agg_config).reset_index()
    
    # Flatten the multi-level columns if needed
    if isinstance(agg_df.columns, pd.MultiIndex):
        agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    
    return agg_df


def aggregate_climate(df_climate: pd.DataFrame,
                      agg_config: Dict[str, Union[str, List[str]]],
                      date_col: str = 'date') -> pd.DataFrame:
    """
    Aggregate the climate dataframe at daily level based on specified aggregation config.
    
    Parameters:
    -----------
    df_climate : pandas DataFrame
        The climate dataframe
    agg_config : dict
        Dictionary with columns as keys and aggregation functions as values
        Example: {'temperature': ['min', 'max', 'mean'], 'humidity': 'mean'}
    date_col : str
        Column name containing the date
    
    Returns:
    --------
    pandas DataFrame
        Aggregated climate dataframe at daily level
    """
    # Make sure the date column is datetime type
    if pd.api.types.is_datetime64_any_dtype(df_climate[date_col]):
        df_climate = df_climate.copy()
    else:
        df_climate = df_climate.copy()
        df_climate[date_col] = pd.to_datetime(df_climate[date_col])
    
    # Group by date, then aggregate
    agg_climate = df_climate.groupby(date_col).agg(agg_config).reset_index()
    
    # Flatten the multi-level columns if needed
    if isinstance(agg_climate.columns, pd.MultiIndex):
        agg_climate.columns = ['_'.join(col).strip('_') for col in agg_climate.columns.values]
    
    return agg_climate


def prepare_training_data(df: pd.DataFrame, 
                          df_climate: pd.DataFrame,
                          unique_dates_sources : pd.DataFrame,
                          df_agg_config: Dict[str, Union[str, List[str]]],
                          climate_agg_config: Dict[str, Union[str, List[str]]],
                          target_col: str = 'kwh_sum',
                          date_col: str = 'date',
                          source_col: str = 'source') -> pd.DataFrame:
    """
    Prepare training data by aggregating both dataframes and merging them.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The main dataframe
    df_climate : pandas DataFrame
        The climate dataframe
    df_agg_config : dict
        Aggregation config for the main dataframe
    climate_agg_config : dict
        Aggregation config for the climate dataframe
    target_col : str
        The target column for prediction (usually a aggregated kwh column)
    date_col : str
        Column name containing the date
    source_col : str
        Column name containing the source identifier
    
    Returns:
    --------
    pandas DataFrame
        Prepared training dataframe with features and target
    """
    # Aggregate the main dataframe
    agg_df = aggregate_df(df, df_agg_config, date_col, source_col)
    
    # Aggregate the climate dataframe
    agg_climate = aggregate_climate(df_climate, climate_agg_config, date_col)
    
    # Merge the aggregated dataframes on date
    training_data = pd.merge(unique_dates_sources, agg_df, on=[date_col, source_col], how='left')

    training_data = pd.merge(training_data, agg_climate, on=date_col, how='left')
    
    # Identify feature columns (all columns except the target)
    feature_cols = [col for col in training_data.columns if col != target_col and col != source_col]
    
    return training_data, feature_cols, target_col, agg_climate


def create_lag_features(df: pd.DataFrame, 
                        cols: List[str], 
                        lags: List[int],
                        date_col: str = 'date',
                        source_col: str = 'source') -> pd.DataFrame:
    """
    Create lag features for specified columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe with daily aggregated data
    cols : list
        List of columns to create lags for
    lags : list
        List of lag values (e.g. [1, 7] for 1-day and 7-day lags)
    date_col : str
        Column name containing the date
    source_col : str
        Column name containing the source identifier
    
    Returns:
    --------
    pandas DataFrame
        Dataframe with added lag features
    """
    # Make a copy to avoid modifying the original dataframe
    df_with_lags = df.copy()
    
    # Ensure the dataframe is sorted by date and source
    df_with_lags = df_with_lags.sort_values([date_col, source_col])
    
    # Create a unique identifier for each source
    for lag in lags:
        for col in cols:
            new_col_name = f"{col}_lag_{lag}"
            
            # Create lag features for each source separately
            df_with_lags[new_col_name] = df_with_lags.groupby(source_col)[col].shift(lag)
    
    # Drop rows with NaN values (beginning of the time series)
    df_with_lags = df_with_lags.dropna()
    
    return df_with_lags


def create_rolling_features(df: pd.DataFrame, 
                           cols: List[str], 
                           windows: List[int],
                           agg_funcs: List[str],
                           date_col: str = 'date',
                           source_col: str = 'source') -> pd.DataFrame:
    """
    Create rolling window features for specified columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe with daily aggregated data
    cols : list
        List of columns to create rolling features for
    windows : list
        List of window sizes (e.g. [3, 7] for 3-day and 7-day windows)
    agg_funcs : list
        List of aggregation functions to apply to rolling windows (e.g. ['mean', 'std'])
    date_col : str
        Column name containing the date
    source_col : str
        Column name containing the source identifier
    
    Returns:
    --------
    pandas DataFrame
        Dataframe with added rolling window features
    """
    # Make a copy to avoid modifying the original dataframe
    df_with_rolling = df.copy()
    
    # Ensure the dataframe is sorted by date and source
    df_with_rolling = df_with_rolling.sort_values([date_col, source_col])
    
    # Create rolling features for each source separately
    for window in windows:
        for col in cols:
            for func in agg_funcs:
                new_col_name = f"{col}_rolling_{window}_{func}"
                
                # Apply rolling window to each source group
                rolling_values = df_with_rolling.groupby(source_col)[col].rolling(window=window).agg(func).reset_index(level=0, drop=True)
                df_with_rolling[new_col_name] = rolling_values
    
    # Drop rows with NaN values (beginning of the time series)
    df_with_rolling = df_with_rolling.dropna()
    
    return df_with_rolling


# Example usage:
if __name__ == "__main__":
    # Load the data
    df = pd.read_pickle('data/cleaned/data.pkl')
    
    # For this example, we'll assume df_climate is available and has certain columns
    # In a real setting, you would load it similarly to df
    # df_climate = pd.read_csv('path_to_climate_data.csv')
    
    # For demonstration purposes, we'll create a mock climate dataframe
    # Remove this in your actual implementation and load your real climate data
    dates = pd.date_range(start='2024-01-01', end='2024-08-01')
    df_climate = pd.DataFrame({
        'date': dates,
        'temperature': np.random.normal(25, 5, len(dates)),
        'humidity': np.random.normal(60, 10, len(dates)),
        'precipitation': np.random.exponential(1, len(dates)),
        'wind_speed': np.random.normal(10, 3, len(dates))
    })
    
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
    
    # Prepare the training data
    training_data, feature_cols, target_col = prepare_training_data(
        df, df_climate, df_agg_config, climate_agg_config
    )
    
    # Add lag features (example: 1-day and 7-day lags for kwh_mean and temperature_mean)
    lag_cols = ['kwh_mean', 'temperature_mean']
    training_data_with_lags = create_lag_features(
        training_data, lag_cols, [1, 7]
    )
    
    # Add rolling window features
    rolling_cols = ['kwh_sum', 'temperature_mean']
    training_data_with_rolling = create_rolling_features(
        training_data, rolling_cols, [3, 7], ['mean', 'std']
    )
    
    print(f"Original df shape: {df.shape}")
    print(f"Training data shape: {training_data.shape}")
    print(f"Training data with lags shape: {training_data_with_lags.shape}")
    print(f"Training data with rolling features shape: {training_data_with_rolling.shape}")
    print(f"Feature columns: {feature_cols}")
    print(f"Target column: {target_col}") 