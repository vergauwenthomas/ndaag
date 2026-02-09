#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial Index Classes

This module defines abstract and concrete financial index classes
for technical analysis of stocks.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd



#ABstract class for indeces
class FinancialIndex(ABC):
    """
    Abstract base class representing a financial/technical index.
    
    This class provides the interface for all financial indices that can be
    calculated from stock data.
    """
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the financial index.
        
        Args:
            name: The name of the index
            parameters: Dictionary of parameters specific to this index
        """
        self.name = name
        self.parameters = parameters or {}
        self._calculated_values = None
        
        
    
    @abstractmethod
    def calculate(self) -> None :
        """
        Calculate the index values from stock data.
        
        Args:
            data: DataFrame containing stock data (OHLCV)
            
        Returns:
            Series containing the calculated index values
        """
        pass
    
    @abstractmethod
    def plot(self):
        """
        Plot the index values.
        
        Args:
            data: DataFrame containing stock data
        """
        if self._calculated_values is None:
            self.calculate(data)
        self._calculated_values.plot(title=f"{self.name} Index")
    
    @abstractmethod
    def find_buy_signal(self) -> pd.DatetimeIndex:
        pass
    
    
    @abstractmethod
    def find_sell_signal(self) -> pd.DatetimeIndex:
        pass
    










