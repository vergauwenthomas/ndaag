#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:40:21 2024

@author: thoverga
"""

import sys
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


from .stock_analysis import StockdAnalysis

class Stock(StockdAnalysis):
    def __init__(self, tickername, fetch_api='yfinance'):
        #General
        self.name = tickername
        self._api = fetch_api
        
        #metadata        
        metadatadict=_fetch_metadata(self.name, self._api)
        self._currency = metadatadict['currency']
        
        #Settings
        self.analysis_stockvalue = 'Close'
    # =============================================================================
    # Specials     
    # =============================================================================
    def __str__(self):
        return f'{self.name} Stock.'

    # def __repr__(self):
        # return 
        
    # =============================================================================
    # Get data    
    # =============================================================================
    def get_data(self, period='1y'):
        df = _fetch_data(tickername=self.name, period=period, api=self._api)
        return df
    
    # =============================================================================
    #  Visuals    
    # =============================================================================
    
    def plot(self, period='3mo', value='Close', *kwargs):
        #Get data
        history_df = _fetch_data(tickername=self.name, period=period, api=self._api)
        #styling attrs    
        title=f'{self.name} at {value}'
        ylabel=self._currency
        # -- make plot ---
        #figure and axes layout
        f = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
        axtop = plt.subplot(gs[0])
        axbot = plt.subplot(gs[1], sharex=axtop)

        #plot to the axes        
        history_df[value].plot(ax=axtop, title=title, ylabel=ylabel)
        history_df['Volume'].plot(ax=axbot, title='Volume')
        
        return axtop
    
    
    
    
    
    
    
    
    

def _fetch_metadata(tickername, api='yfinance'):
    metadict = {'currency': None}
    if api=='yfinance':
        metadict['currency'] = yf.Ticker(str(tickername).upper()).fast_info['currency']
    else:
        sys.error(f'API: {api} is not implemented.')
    
    return metadict



def _fetch_data(tickername, period, api='yfinance'):
    if api=='yfinance':
        return yf.Ticker(str(tickername).upper()).history(period=period)
    else:
        sys.error(f'API: {api} is not implemented.')
        
    
    