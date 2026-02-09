
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ndaag.indeces.abstractclass import FinancialIndex


class CCIIndex(FinancialIndex):
    """
    CCI (Commodity Channel Index) for a Stock instance.
    
    The CCI is a momentum-based oscillator used to help determine when an 
    investment vehicle is reaching a condition of being overbought or oversold.
    
    Formula:
        CCI = (Typical Price - SMA of Typical Price) / (0.015 × Mean Deviation)
    
    Where:
        - Typical Price = (High + Low + Close) / 3
        - SMA = Simple Moving Average of Typical Price
        - Mean Deviation = Mean absolute deviation from the SMA
        - 0.015 = Constant to ensure ~70-80% of values fall between -100 and +100
    
    Interpretation:
        - CCI > +100: Overbought condition (potential sell opportunity)
        - CCI < -100: Oversold condition (potential buy opportunity)
        - CCI crossing above -100: Buy signal
        - CCI crossing below +100: Sell signal
    """
    
    def __init__(self, stock, period=20, constant=0.015):
        """
        Initialize the CCI index.
        
        Args:
            stock: Stock instance to calculate CCI for
            period: Number of periods for moving average (default: 20)
            constant: Lambert constant for scaling (default: 0.015)
        """
        parameters = {
            'period': period,
            'constant': constant
        }
        super().__init__(name='CCI', parameters=parameters)
        
        # Stock
        self.stock = stock
        
        # Index values
        self.cci = pd.Series(dtype=float)
        self.typical_price = pd.Series(dtype=float)
        
        # Signals
        self.buy_signals = pd.DatetimeIndex([])
        self.sell_signals = pd.DatetimeIndex([])
    
    
    def calculate(self):
        """
        Calculate the CCI (Commodity Channel Index).
        
        Steps:
        1. Calculate Typical Price = (High + Low + Close) / 3
        2. Calculate SMA of Typical Price over the period
        3. Calculate Mean Deviation
        4. Calculate CCI = (Typical Price - SMA) / (constant × Mean Deviation)
        """
        data = self.stock.data.copy()
        period = self.parameters['period']
        constant = self.parameters['constant']
        
        # Calculate Typical Price
        self.typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        self.typical_price.name = 'Typical Price'
        
        # Calculate Simple Moving Average of Typical Price
        sma_tp = self.typical_price.rolling(window=period, min_periods=period).mean()
        
        # Calculate Mean Deviation
        # Mean Deviation = Mean of |Typical Price - SMA|
        def calculate_mean_deviation(window_data):
            """Calculate mean absolute deviation for a window."""
            mean = window_data.mean()
            mad = (window_data - mean).abs().mean()
            return mad
        
        mean_deviation = self.typical_price.rolling(
            window=period, 
            min_periods=period
        ).apply(calculate_mean_deviation, raw=False)
        
        # Calculate CCI
        # Avoid division by zero
        self.cci = (self.typical_price - sma_tp) / (constant * mean_deviation)
        self.cci.name = 'CCI'
        
        # Replace inf values with NaN
        self.cci.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    
    def plot(self, show_buy_signals: bool = False, 
             show_sell_signals: bool = False,
             figsize=(12, 6),
             **plotkwargs):
        """
        Plot the CCI indicator with overbought/oversold levels.
        
        Args:
            show_buy_signals: If True, show vertical lines at buy signals
            show_sell_signals: If True, show vertical lines at sell signals
            figsize: Figure size as (width, height)
            **plotkwargs: Additional plotting arguments
        
        Returns:
            matplotlib axes object
        """
        if self.cci.empty:
            raise ValueError("CCI not calculated. Call calculate() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot CCI line
        ax.plot(self.cci.index, self.cci.values, 
                label=f'CCI ({self.parameters["period"]})', 
                color='blue', linewidth=1.5)
        
        # Add overbought/oversold lines at +100 and -100
        ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, 
                   alpha=0.7, label='Overbought (+100)')
        ax.axhline(y=-100, color='green', linestyle='--', linewidth=1.5, 
                   alpha=0.7, label='Oversold (-100)')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Shade overbought/oversold regions
        ax.axhspan(100, ax.get_ylim()[1], alpha=0.1, color='red')
        ax.axhspan(-100, ax.get_ylim()[0], alpha=0.1, color='green')
        
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
        ax.set_title(f"CCI - Commodity Channel Index (Period: {self.parameters['period']})", 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('CCI Value', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return ax
    
    
    def find_buy_signal(self, threshold: float = -100.0) -> pd.DatetimeIndex:
        """
        Find buy signals based on CCI analysis.
        
        Buy signals are generated when CCI crosses above the threshold from below,
        indicating the stock is moving from oversold to neutral/bullish territory.
        
        Args:
            threshold: CCI level for buy signal (default: -100.0)
                      Standard interpretation: crossing above -100 suggests
                      the oversold condition is ending.
        
        Returns:
            DatetimeIndex of timestamps where buy signals occur
        """
        if self.cci.empty:
            raise ValueError("CCI not calculated. Call calculate() first.")
        
        # CCI above threshold
        above_threshold = self.cci > threshold
        
        # Find where it crosses from below to above threshold
        # (was below, now above)
        crossed_above = above_threshold & ~above_threshold.shift(1, fill_value=False)
        
        self.buy_signals = self.cci[crossed_above].index
        return self.buy_signals
    
    
    def find_sell_signal(self, threshold: float = 100.0) -> pd.DatetimeIndex:
        """
        Find sell signals based on CCI analysis.
        
        Sell signals are generated when CCI crosses below the threshold from above,
        indicating the stock is moving from overbought to neutral/bearish territory.
        
        Args:
            threshold: CCI level for sell signal (default: 100.0)
                      Standard interpretation: crossing below +100 suggests
                      the overbought condition is ending.
        
        Returns:
            DatetimeIndex of timestamps where sell signals occur
        """
        if self.cci.empty:
            raise ValueError("CCI not calculated. Call calculate() first.")
        
        # CCI below threshold
        below_threshold = self.cci < threshold
        
        # Find where it crosses from above to below threshold
        # (was above, now below)
        crossed_below = below_threshold & ~below_threshold.shift(1, fill_value=False)
        
        self.sell_signals = self.cci[crossed_below].index
        return self.sell_signals
