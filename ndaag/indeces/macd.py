
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ndaag.indeces.abstractclass import FinancialIndex
from ndaag.modules.signal_processing import calculate_derivative, calculate_ema

class MACDIndex(FinancialIndex):
    """
    MACD (Moving Average Convergence Divergence) index for a Stock instance.
    
    The MACD is a trend-following momentum indicator that shows the relationship
    between two exponential moving averages (EMAs) of a stock's price.
    
    Components:
    - MACD Line: 12-day EMA minus 26-day EMA
    - Signal Line: 9-day EMA of the MACD line
    - Histogram: MACD line minus Signal line
    
    Buy/Sell signals are generated when the MACD line crosses the Signal line.
    """
    
    def __init__(self, stock, ema_fast=12, ema_slow=26, ema_signal=9):
        """
        Initialize the MACD index.
        
        Args:
            stock: Stock instance to calculate MACD for
            ema_fast: Period for fast EMA (default: 12)
            ema_slow: Period for slow EMA (default: 26)
            ema_signal: Period for signal line EMA (default: 9)
        """
        parameters = {
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'ema_signal': ema_signal
        }
        super().__init__(name='MACD', parameters=parameters)
        
        # Stock
        self.stock = stock
        
        # Index values
        self.macd_line = pd.Series(dtype=float)
        self.signal_line = pd.Series(dtype=float)
        self.histogram = pd.Series(dtype=float)
        
        # Signals
        self.buy_signals = pd.DatetimeIndex([])
        self.sell_signals = pd.DatetimeIndex([])
    
    
    def calculate(self):
        """
        Calculate the MACD indicator components.
        
        Calculates:
        - MACD line: 12-day EMA - 26-day EMA
        - Signal line: 9-day EMA of MACD line
        - Histogram: MACD line - Signal line
        """
        close_prices = self.stock.data['Close'].copy()
        
        # Calculate the two EMAs
        ema_fast = calculate_ema(close_prices, window=self.parameters['ema_fast'])
        ema_slow = calculate_ema(close_prices, window=self.parameters['ema_slow'])
        
        # MACD line is the difference between fast and slow EMAs
        self.macd_line = ema_fast - ema_slow
        self.macd_line.name = 'MACD'
        
        # Signal line is the EMA of the MACD line
        self.signal_line = calculate_ema(self.macd_line.dropna(), 
                                         window=self.parameters['ema_signal'])
        self.signal_line.name = 'Signal'
        
        # Histogram is the difference between MACD and Signal
        self.histogram = self.macd_line - self.signal_line
        self.histogram.name = 'Histogram'
    
    
    def plot(self, show_buy_signals: bool = False, 
             show_sell_signals: bool = False,
             figsize=(12, 6),
             **plotkwargs):
        """
        Plot the MACD indicator with MACD line, Signal line, and Histogram.
        
        Args:
            show_buy_signals: If True, show vertical lines at buy signals
            show_sell_signals: If True, show vertical lines at sell signals
            figsize: Figure size as (width, height)
            **plotkwargs: Additional plotting arguments
        
        Returns:
            matplotlib axes object
        """
        if self.macd_line.empty:
            raise ValueError("MACD not calculated. Call calculate() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot MACD and Signal lines
        ax.plot(self.macd_line.index, self.macd_line.values, 
                label='MACD Line', color='blue', linewidth=1.5)
        ax.plot(self.signal_line.index, self.signal_line.values, 
                label='Signal Line', color='red', linewidth=1.5)
        
        # Plot histogram as bars
        colors = ['green' if val >= 0 else 'red' for val in self.histogram.values]
        ax.bar(self.histogram.index, self.histogram.values, 
               label='Histogram', color=colors, alpha=0.3, width=0.8)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Add buy/sell signals if requested
        if show_buy_signals and len(self.buy_signals) > 0:
            ax.vlines(x=self.buy_signals, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                      colors='green', linestyle='-', alpha=0.6, linewidth=3,
                      label='Buy Signal')
        
        if show_sell_signals and len(self.sell_signals) > 0:
            ax.vlines(x=self.sell_signals, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                      colors='red', linestyle='-', alpha=0.6, linewidth=3,
                      label='Sell Signal')
        
        # Formatting
        ax.set_title(f"MACD ({self.parameters['ema_fast']}, "
                     f"{self.parameters['ema_slow']}, "
                     f"{self.parameters['ema_signal']})", 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('MACD Value', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return ax
    
    
    def find_buy_signal(self, early_warning_threshold: float = 0.0,
                        smooth_window: int = 3) -> pd.DatetimeIndex:
        """
        Find buy signals based on MACD analysis.
        
        Buy signals are generated when the MACD line crosses above the Signal line
        (histogram goes from negative to positive).
        
        With early_warning_threshold > 0, signals are triggered before the actual
        crossover when:
        1. Histogram is negative but close to zero (within threshold)
        2. Histogram momentum (derivative) is positive (rising toward zero)
        
        Args:
            early_warning_threshold: Distance from zero to trigger early warnings.
                                    Set to 0.0 for exact crossovers only (default).
                                    Higher values give earlier but potentially more
                                    false signals. Typical range: 0.1 to 1.0.
            smooth_window: Window size for calculating histogram momentum (default: 3)
        
        Returns:
            DatetimeIndex of timestamps where buy signals occur
            
        Example:
            >>> # Exact crossovers only
            >>> buy_signals = macd.find_buy_signal()
            >>> # Early warnings when histogram is within 0.5 of zero
            >>> early_buy = macd.find_buy_signal(early_warning_threshold=0.5)
        """
        if self.histogram.empty:
            raise ValueError("MACD not calculated. Call calculate() first.")
        
        if early_warning_threshold == 0.0:
            # Original behavior: exact crossover detection
            histogram_positive = self.histogram > 0
            crossed_above = histogram_positive & ~histogram_positive.shift(1, fill_value=False)
            buy_signal = crossed_above
        else:
            # Early warning mode: detect approaching crossover
            # Calculate histogram momentum (rate of change)
            histogram_momentum = calculate_derivative(
                self.histogram,
                smooth_window=smooth_window,
                smooth_method='rolling'
            )
            
            # Condition 1: Histogram is negative (below zero)
            is_negative = self.histogram < 0
            
            # Condition 2: Histogram is close to zero (within threshold)
            is_near_zero = self.histogram.abs() < early_warning_threshold
            
            # Condition 3: Histogram is rising (positive momentum)
            is_rising = histogram_momentum > 0
            
            # Combine conditions for early warning
            early_warning = is_negative & is_near_zero & is_rising
            
            # Condition 4: Actual crossover (for completeness)
            histogram_positive = self.histogram > 0
            crossed_above = histogram_positive & ~histogram_positive.shift(1, fill_value=False)
            
            # Trigger on either early warning OR actual crossover
            # But avoid duplicate signals by only triggering once
            buy_signal = early_warning | crossed_above
            
            # Remove consecutive signals (keep only first in a series)
            buy_signal = buy_signal & ~buy_signal.shift(1, fill_value=False)
        
        self.buy_signals = self.histogram[buy_signal].index
        return self.buy_signals
    
    
    def find_sell_signal(self, early_warning_threshold: float = 0.0,
                         smooth_window: int = 3) -> pd.DatetimeIndex:
        """
        Find sell signals based on MACD analysis.
        
        Sell signals are generated when the MACD line crosses below the Signal line
        (histogram goes from positive to negative).
        
        With early_warning_threshold > 0, signals are triggered before the actual
        crossover when:
        1. Histogram is positive but close to zero (within threshold)
        2. Histogram momentum (derivative) is negative (falling toward zero)
        
        Args:
            early_warning_threshold: Distance from zero to trigger early warnings.
                                    Set to 0.0 for exact crossovers only (default).
                                    Higher values give earlier but potentially more
                                    false signals. Typical range: 0.1 to 1.0.
            smooth_window: Window size for calculating histogram momentum (default: 3)
        
        Returns:
            DatetimeIndex of timestamps where sell signals occur
            
        Example:
            >>> # Exact crossovers only
            >>> sell_signals = macd.find_sell_signal()
            >>> # Early warnings when histogram is within 0.5 of zero
            >>> early_sell = macd.find_sell_signal(early_warning_threshold=0.5)
        """
        if self.histogram.empty:
            raise ValueError("MACD not calculated. Call calculate() first.")
        
        if early_warning_threshold == 0.0:
            # Original behavior: exact crossover detection
            histogram_negative = self.histogram < 0
            crossed_below = histogram_negative & ~histogram_negative.shift(1, fill_value=False)
            sell_signal = crossed_below
        else:
            # Early warning mode: detect approaching crossover
            # Calculate histogram momentum (rate of change)
            histogram_momentum = calculate_derivative(
                self.histogram,
                smooth_window=smooth_window,
                smooth_method='rolling'
            )
            
            # Condition 1: Histogram is positive (above zero)
            is_positive = self.histogram > 0
            
            # Condition 2: Histogram is close to zero (within threshold)
            is_near_zero = self.histogram.abs() < early_warning_threshold
            
            # Condition 3: Histogram is falling (negative momentum)
            is_falling = histogram_momentum < 0
            
            # Combine conditions for early warning
            early_warning = is_positive & is_near_zero & is_falling
            
            # Condition 4: Actual crossover (for completeness)
            histogram_negative = self.histogram < 0
            crossed_below = histogram_negative & ~histogram_negative.shift(1, fill_value=False)
            
            # Trigger on either early warning OR actual crossover
            # But avoid duplicate signals by only triggering once
            sell_signal = early_warning | crossed_below
            
            # Remove consecutive signals (keep only first in a series)
            sell_signal = sell_signal & ~sell_signal.shift(1, fill_value=False)
        
        self.sell_signals = self.histogram[sell_signal].index
        return self.sell_signals