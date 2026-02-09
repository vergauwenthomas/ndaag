#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Processing Utilities

This module provides functions for processing time series data,
including derivative calculations and smoothing operations.
"""

import pandas as pd
import numpy as np


def calculate_derivative(
    series: pd.Series,
    smooth_window: int = 3,
    smooth_method: str = 'rolling',
    center: bool = True
) -> pd.Series:
    """
    Calculate the time derivative of a discrete time series signal.
    
    The derivative is computed as the rate of change between consecutive
    time points. Optional smoothing can be applied before or after
    derivative calculation to reduce noise.
    
    Args:
        series: Input pandas Series with DatetimeIndex or numeric index
        smooth_window: Window size for smoothing operation (default: 3)
                      Set to 1 or None to disable smoothing
        smooth_method: Smoothing method to use:
                      - 'rolling': Simple moving average (default)
                      - 'ewm': Exponential weighted moving average
                      - 'savgol': Savitzky-Golay filter (requires scipy)
        center: If True, centers the rolling window for smoothing (default: True)
                Only applies to 'rolling' method
    
    Returns:
        pandas Series with the same index as input, containing derivative values
        The Series name will be preserved from the input
    
    Example:
        >>> data = pd.Series([1, 2, 4, 7, 11], 
        ...                  index=pd.date_range('2024-01-01', periods=5))
        >>> derivative = calculate_derivative(data, smooth_window=2)
    """
    # Validate input
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    
    if len(series) < 2:
        raise ValueError("Series must have at least 2 data points")
    
    # Apply smoothing to input signal if requested
    if smooth_window and smooth_window > 1:
        smoothed = _apply_smoothing(series, smooth_window, smooth_method, center)
    else:
        smoothed = series.copy()
    
    # Calculate time differences (dt)
    if isinstance(series.index, pd.DatetimeIndex):
        # Convert time differences to days
        dt = series.index.to_series().diff().dt.total_seconds() / 86400.0
        # Avoid division by zero
        dt = dt.replace(0, np.nan)
    else:
        # Assume uniform spacing for numeric indices
        dt = series.index.to_series().diff()
        dt = dt.replace(0, np.nan)
    
    # Calculate value differences (dy)
    dy = smoothed.diff()
    
    # Calculate derivative: dy/dt
    derivative = dy / dt
    
    # Create result series with same name and index
    result = pd.Series(
        derivative.values,
        index=series.index,
        name=series.name
    )
    
    return result


def _apply_smoothing(
    series: pd.Series,
    window: int,
    method: str = 'rolling',
    center: bool = True
) -> pd.Series:
    """
    Apply smoothing to a time series.
    
    Args:
        series: Input pandas Series
        window: Window size for smoothing
        method: Smoothing method ('rolling', 'ewm', or 'savgol')
        center: Whether to center the rolling window
    
    Returns:
        Smoothed pandas Series
    """
    if method == 'rolling':
        # Simple moving average
        smoothed = series.rolling(
            window=window,
            center=center,
            min_periods=1
        ).mean()
        
    elif method == 'ewm':
        # Exponential weighted moving average
        smoothed = series.ewm(
            span=window,
            adjust=False,
            min_periods=1
        ).mean()
        
    elif method == 'savgol':
        # Savitzky-Golay filter (requires scipy)
        try:
            from scipy.signal import savgol_filter
            # Ensure window is odd
            window = window if window % 2 == 1 else window + 1
            # Polynomial order (typically 2 or 3)
            polyorder = min(2, window - 1)
            
            smoothed_values = savgol_filter(
                series.values,
                window_length=window,
                polyorder=polyorder,
                mode='nearest'
            )
            smoothed = pd.Series(smoothed_values, index=series.index, name=series.name)
            
        except ImportError:
            raise ImportError(
                "scipy is required for 'savgol' smoothing method. "
                "Install it with: pip install scipy"
            )
    else:
        raise ValueError(
            f"Unknown smoothing method: {method}. "
            "Choose from: 'rolling', 'ewm', 'savgol'"
        )
    
    return smoothed


def calculate_ema(
    series: pd.Series,
    window: int,
    adjust: bool = False
) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA) of a time series.
    
    The EMA is a type of moving average that places a greater weight and 
    significance on the most recent data points. The weighting for older 
    data points decreases exponentially, never reaching zero.
    
    EMA formula:
        EMA_today = (Value_today * (smoothing / (1 + window))) + 
                    (EMA_yesterday * (1 - (smoothing / (1 + window))))
    
    Where smoothing = 2 (standard value for EMA)
    
    Args:
        series: Input pandas Series with time series data
        window: Number of periods for the EMA calculation (e.g., 9, 12, 26)
        adjust: If True, uses the adjusted formula which divides by decaying 
                adjustment factor in beginning periods. Default is False to 
                match standard EMA calculation.
    
    Returns:
        pandas Series with the same index as input, containing EMA values
        The first (window-1) values will be NaN unless adjust=True
    
    Example:
        >>> data = pd.Series([22, 24, 23, 25, 27, 26, 28, 30, 29])
        >>> ema_9 = calculate_ema(data, window=9)
        >>> ema_12 = calculate_ema(data, window=12)
        >>> ema_26 = calculate_ema(data, window=26)
    """
    # Validate input
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    
    if window < 1:
        raise ValueError("Window must be at least 1")
    
    if len(series) < window:
        raise ValueError(f"Series length ({len(series)}) must be at least equal to window ({window})")
    
    # Calculate EMA using pandas ewm (exponentially weighted moving average)
    ema = series.ewm(
        span=window,
        adjust=adjust,
        min_periods=window
    ).mean()
    
    # Preserve the series name
    ema.name = f"EMA_{window}" if series.name is None else f"{series.name}_EMA_{window}"
    
    return ema


def calculate_multiple_emas(
    series: pd.Series,
    windows: list = None,
    adjust: bool = False
) -> pd.DataFrame:
    """
    Calculate multiple Exponential Moving Averages (EMAs) for different time windows.
    
    This is a convenience function for calculating several EMAs at once, commonly
    used in technical analysis (e.g., 9-day, 12-day, and 26-day EMAs for MACD).
    
    Args:
        series: Input pandas Series with time series data
        windows: List of window sizes for EMA calculation. 
                 Default is [9, 12, 26] (common for MACD indicator)
        adjust: If True, uses the adjusted formula. Default is False.
    
    Returns:
        pandas DataFrame with columns for each EMA, indexed same as input series
        Column names will be 'EMA_9', 'EMA_12', 'EMA_26', etc.
    
    Example:
        >>> data = pd.Series([22, 24, 23, 25, 27, 26, 28, 30, 29, 31, 30, 32])
        >>> emas = calculate_multiple_emas(data, windows=[9, 12, 26])
        >>> # Access individual EMAs
        >>> ema_9 = emas['EMA_9']
        >>> ema_12 = emas['EMA_12']
    """
    if windows is None:
        windows = [9, 12, 26]
    
    # Validate input
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    
    if not isinstance(windows, (list, tuple)):
        raise TypeError("Windows must be a list or tuple of integers")
    
    if len(windows) == 0:
        raise ValueError("At least one window size must be specified")
    
    # Calculate EMA for each window
    emas = {}
    for window in windows:
        ema = calculate_ema(series, window=window, adjust=adjust)
        emas[f"EMA_{window}"] = ema
    
    # Combine into DataFrame
    result = pd.DataFrame(emas, index=series.index)
    
    return result
