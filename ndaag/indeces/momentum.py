import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ndaag.indeces.abstractclass import FinancialIndex
from ndaag.modules.signal_processing import calculate_derivative, calculate_ema


class MomentumIndex(FinancialIndex):
    """
    Momentum index for a Stock instance.
    
    Measures the rate of change in stock prices over a specified period,
    helping identify the strength and direction of price trends.
    """
    
    def __init__(self, stock, window='10d', interpolate_weekend=True):
        """
        Initialize the momentum index.
        
        Args:
            stock: Stock instance to calculate momentum for
            historysize: Size of the historical data to consider
            window: Window size for calculating momentum
        """
        parameters = {
            'window': window,
            'interpolate_weekend': interpolate_weekend
        }
        super().__init__(name='Momentum', parameters=parameters)
        
        #Stock
        self.stock = stock
        
        #Index values
        self.momentum = pd.Series(dtype=float)
    
        #Signals
        self.buy_signals = pd.DatetimeIndex([])
        self.sell_signals = pd.DatetimeIndex([])
    
    
    def calculate(self):
        #shift is only relevant on day scale, the time the market closed is irrelevant
        records = self.stock.data[['Close']].copy()
        records['ref'] = records['Close'].shift(freq=self.parameters['window'])
        records['momentum'] = records['Close'] - records['ref']

        #because of weekends, there is often no reference availabel resulting in Nan
        # we can interpolate (simple linear) these missing momentums
        if self.parameters['interpolate_weekend']:
            records['momentum'].interpolate(method='linear', limit=2, inplace=True)
        self.momentum = records['momentum']
        

    def plot(self, show_buy_signals: bool = False, 
             show_sell_signals: bool = False,
             **plotkwargs):
        ax = self.momentum.plot(title=f"{self.parameters['window']} {self.name} Index", **plotkwargs, ylabel='Momentum')
        ax.axhline(y=0, color='black', linestyle='--')
        
        if show_buy_signals:
           ax.vlines(x=self.buy_signals, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                     colors='green', linestyle='-',
                     alpha = 0.4, linewidth=5,
                     label='Buy Signal'
                     )
        
        if show_sell_signals:
           ax.vlines(x=self.sell_signals, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors='red',
                     linestyle='-', alpha = 0.4, linewidth=5,
                     label='Sell Signal')

        plt.legend()

        return ax
        
    
    def find_buy_signal(self, threshold: float = 0.0, 
                        require_rising: bool = True,
                        smooth_window: int = 3) -> pd.DatetimeIndex:
        """
        Find buy signals based on momentum analysis.
        
        Buy signals are generated when:
        1. Momentum crosses above the threshold (from below)
        2. Optionally: Momentum is rising (positive derivative)
        
        Args:
            threshold: Momentum threshold for buy signal (default: 0.0)
            require_rising: If True, momentum must be rising (default: True)
            smooth_window: Smoothing window for derivative calculation (default: 3)
        
        Returns:
            DatetimeIndex of timestamps where buy signals occur
        """
        if self.momentum.empty:
            raise ValueError("Momentum not calculated. Call calculate() first.")
        
        # Condition 1: Momentum above threshold
        above_threshold = self.momentum >= threshold
        
        # Condition 2: Momentum crosses from below to above threshold
        # (regime change: was below, now above)
        crossed_above = above_threshold & ~above_threshold.shift(1, fill_value=False)
        
        # Condition 3 (optional): Momentum is rising
        if require_rising:
            momentum_derivative = calculate_derivative(
                self.momentum, 
                smooth_window=smooth_window,
                smooth_method='rolling'
            )
            is_rising = momentum_derivative > 0
            
            # Combine conditions
            buy_signal = crossed_above & is_rising
        else:
            buy_signal = crossed_above
        
        self.buy_signals = self.momentum[buy_signal].index
        # Return timestamps where buy signal is True
        return self.momentum[buy_signal].index
        
    
    def find_sell_signal(self, threshold: float = 0.0,
                         require_falling: bool = True,
                         smooth_window: int = 3) -> pd.DatetimeIndex:
        """
        Find sell signals based on momentum analysis.
        
        Sell signals are generated when:
        1. Momentum crosses below the threshold (from above)
        2. Optionally: Momentum is falling (negative derivative)
        
        Args:
            threshold: Momentum threshold for sell signal (default: 0.0)
            require_falling: If True, momentum must be falling (default: True)
            smooth_window: Smoothing window for derivative calculation (default: 3)
        
        Returns:
            DatetimeIndex of timestamps where sell signals occur
        """
        if self.momentum.empty:
            raise ValueError("Momentum not calculated. Call calculate() first.")
        
        # Condition 1: Momentum below threshold
        below_threshold = self.momentum <= threshold
        
        # Condition 2: Momentum crosses from above to below threshold
        # (regime change: was above, now below)
        crossed_below = below_threshold & ~below_threshold.shift(1, fill_value=False)
        
        # Condition 3 (optional): Momentum is falling
        if require_falling:
            momentum_derivative = calculate_derivative(
                self.momentum,
                smooth_window=smooth_window,
                smooth_method='rolling'
            )
            is_falling = momentum_derivative < 0
            
            # Combine conditions
            sell_signal = crossed_below & is_falling
        else:
            sell_signal = crossed_below
        
        self.sell_signals = self.momentum[sell_signal].index
        # Return timestamps where sell signal is True
        return self.sell_signals
      
