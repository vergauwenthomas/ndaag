#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:16:13 2024

@author: thoverga
"""

import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import sys

from ndaag.stock import Stock


class StockdAnalysis():

    def __init__(self, stock: Stock):
        #General
        self.stock = stock
    
    
    def _plot_event_analysis(self, event, historysize, eventdf):
        #Get data
        data = self.stock.fetch_data(period=historysize)[[self.analysis_stockvalue, 'Volume']]
        
        #Get Indices
        indices_df_list = []
        #macd
        indices_df_list.append(self.get_MACD_and_signal(historysize=historysize))
        #momentum
        indices_df_list.append(self.get_momentum(historysize=historysize))
        #rsi
        indices_df_list.append(self.get_RSI(historysize=historysize))
        #rol_avg
        indices_df_list.append(self.get_rolling_average(historysize=historysize))
        
        for index in indices_df_list:
            data = data.merge(index,
                              how='left',
                              left_index=True,
                              right_index=True)
        
        #make grid
        #figure and axes layout
        f = plt.figure()
        gs = gridspec.GridSpec(5, 1, height_ratios=[4,1,1,1,1])
        
        #1. top plot --> value
        axtop = plt.subplot(gs[0])
        data[self.analysis_stockvalue].plot(ax=axtop,
                                            title=f'{self.name} at {self.analysis_stockvalue}',
                                            ylabel=self._currency)
        #add roling avg
        data['rol_avg'].plot(ax=axtop, ls='--')
        
        buy_triggers = eventdf[eventdf['buy-trigger']].index.to_list()
        (ymin, ymax) = axtop.get_ylim()
        axtop.vlines(x=buy_triggers,
                     ymin=ymin, ymax=ymax,
                     colors='purple',
                     ls='--', lw=2,
                     label='Buy triggers')
        
        
        #2.  volume        
        ax_volume = plt.subplot(gs[1], sharex=axtop)
        data['Volume'].plot(ax=ax_volume, title='Volume')
        
        #3 MACD 
        ax_MACD = plt.subplot(gs[2], sharex=axtop)
        data[['MACD', 'MACD_signal']].plot(ax=ax_MACD, title='MACD and signal')
        
        column = 'MACD_buy_signal'
        (ymin, ymax) = ax_MACD.get_ylim()
        ax_MACD.vlines(x=eventdf[eventdf[column]].index.to_list(),
                     ymin=ymin, ymax=ymax,
                     colors='purple',
                     ls='--', lw=2,
                     label=column)
        
        #4 momentum 
        ax_momentum = plt.subplot(gs[3], sharex=axtop)
        ax_momentum.axhline(y=0, ls='--', color='black')
        data[['momentum']].plot(ax=ax_momentum, title='Momentum')
        
        column = 'momentum_buy_trigger'
        (ymin, ymax) = ax_momentum.get_ylim()
        ax_momentum.vlines(x=eventdf[eventdf[column]].index.to_list(),
                     ymin=ymin, ymax=ymax,
                     colors='purple',
                     ls='--', lw=2,
                     label=column)
        
        
        #5 RSI
        ax_rsi = plt.subplot(gs[4], sharex=axtop)
        ax_rsi.axhline(y=70, ls='--', color='black')
        ax_rsi.axhline(y=40, ls='--', color='black')
        data[['rsi']].plot(ax=ax_rsi, title='RSI')
        
        column = 'rsi_buy_trigger'
        (ymin, ymax) = ax_rsi.get_ylim()
        ax_rsi.vlines(x=eventdf[eventdf[column]].index.to_list(),
                     ymin=ymin, ymax=ymax,
                     colors='purple',
                     ls='--', lw=2,
                     label=column)
        
        
        return
                
        
        
    
    def event_analysis(self, make_plot=True, event='buy', historysize='1y', momentum_smoothing='2d',
                       rsi_smoothing='2d', rsi_buy_threshold=40.):
        
        #Create empty df with datetie idx related to history size
        #TODO: construct from historysize directly (and drop weekenddays?)
        datadf = self.get_data(period=historysize)
        signaldf = pd.DataFrame()
        signaldf.index = datadf.index.copy()
       
        
        # ------ MACD event signal ---------
        macdf = self.get_MACD_and_signal(historysize=historysize)
        if event =='buy':
            #buy conditions: mac larger than signal and mac just crossed signal?
            #cond1: macd greater then signal
            macdf['cond1'] = (macdf['MACD'] >= macdf['MACD_signal'])
            
            #cond2: change of regime (lines are crossing)
            macdf['regime'] = macdf['cond1'] #True if macd over signal, false if macd under signal
            macdf['cond2'] =  (macdf['regime'] != macdf['regime'].shift())   
            
           
            macdf['MACD_buy_signal'] = macdf['cond1'] & macdf['cond2']
            signaldf = signaldf.merge(macdf[['MACD_buy_signal']],
                                      how='left',
                                      left_index=True,
                                      right_index=True)
        else:
            sys.exit(f'{event} not implemented')
            
        
        # ------ Momentum event signal ------
        momentumdf = self.get_momentum(historysize=historysize)
        #Smooth momentum signal
        momentumdf['smoothed_momentum'] = momentumdf['momentum'].rolling(window=momentum_smoothing).mean()

        if event == 'buy':
            #buy conditions: positive momentum and increasing in the last 2 days?
            #momentum must be positive
            momentumdf['cond1'] = momentumdf['smoothed_momentum'] > 0
            #mementum is increasing over the last 2 days
            momentumdf['cond2'] = ((momentumdf['smoothed_momentum'] > momentumdf['smoothed_momentum'].shift(1)) &
                                   (momentumdf['smoothed_momentum'].shift(1) > momentumdf['smoothed_momentum'].shift(2)))

            momentumdf['momentum_buy_trigger'] = momentumdf['cond1'] & momentumdf['cond2']
            

            signaldf = signaldf.merge(momentumdf[['momentum_buy_trigger']],
                                      how='left',
                                      left_index=True,
                                      right_index=True)
        else:
            sys.exit(f'{event} not implemented')


    # ------ RSI event signal ------
    

        rsidf = self.get_RSI(historysize=historysize)
            
        #Smooth signal
        rsidf['smoothed_rsi'] = rsidf['rsi'].rolling(window=rsi_smoothing).mean()
        
        if event == 'buy':
            #buy conditions: positive momentum and increasing in the last 2 days?
            #rsi smaller than threshold
            rsidf['cond1'] = rsidf['smoothed_rsi'] >= rsi_buy_threshold
            #rsi is increasing over the last 2 days
            rsidf['cond2'] = ((rsidf['smoothed_rsi'] > rsidf['smoothed_rsi'].shift(1)) &
                                   (rsidf['smoothed_rsi'].shift(1) > rsidf['smoothed_rsi'].shift(2)))
        
            rsidf['rsi_buy_trigger'] = rsidf['cond1'] & rsidf['cond2']
            
        
            signaldf = signaldf.merge(rsidf[['rsi_buy_trigger']],
                                      how='left',
                                      left_index=True,
                                      right_index=True)
        else:
            sys.exit(f'{event} not implemented')
            
        
        
        
        
        #define tiggers
        if event=='buy':
            signaldf['buy-trigger'] = signaldf[['MACD_buy_signal',
                                    'momentum_buy_trigger',
                                    'rsi_buy_trigger']].apply(np.all, axis='columns')
        
        if make_plot:
            self._plot_event_analysis(event=event,
                                      historysize=historysize,
                                      eventdf=signaldf)
        
        
        return signaldf
    


        
       
    
    def get_rolling_average(self, historysize='1y', rollingwindowsize='4d', trg_series_name='rol_avg'):
        records = self.get_data(historysize)[self.analysis_stockvalue].to_frame()
        rol_avg = records.rolling( 
                        window='4d',
                        min_periods=2, #min windwsize
                        center=False,
                        win_type=None,
                        on=None,
                        axis='index',
                        closed=None,
                        step=None,
                        method='single').mean()
        rol_avg=rol_avg.rename(columns={self.analysis_stockvalue: trg_series_name})
        return rol_avg
    
    
    def get_MACD_and_signal(self, historysize='1y', short_period_ema=12, long_period_ema=26,
                            signal_period_ema=9, trg_series_MACD_name='MACD', trg_series_signal_name='MACD_signal'):
        records = self.get_data(historysize)[self.analysis_stockvalue].to_frame()
        
        # Calculate the 12-period EMA
        records['EMA12'] = records[self.analysis_stockvalue].ewm(span=12, adjust=False).mean()
        
        # Calculate the 26-period EMA
        records['EMA26'] = records[self.analysis_stockvalue].ewm(span=26, adjust=False).mean()
        
        # Calculate MACD (the difference between 12-period EMA and 26-period EMA)
        records[trg_series_MACD_name] = records['EMA12'] - records['EMA26']
        
        # Calculate the 9-period EMA of MACD (Signal Line)
        records[trg_series_signal_name] = records[trg_series_MACD_name].ewm(span=9, adjust=False).mean()

        return records[[trg_series_MACD_name, trg_series_signal_name]]


    def get_momentum(self, historysize='1y', price_change_period='10d',
                     interpolate_over_weekends=True, trg_series_name='momentum'):
        records = self.get_data(historysize)[self.analysis_stockvalue].to_frame()
        
        #shift is only relevant on day scale, the time the market closed is irrelevant
        records = records.reset_index()
        records['just_date'] = records['Date'].dt.date
        records = records.set_index(pd.to_datetime(records['just_date']))
        #calculate value change over the period
        records['ref'] = records['Close'].shift(freq=price_change_period)
        records['diff'] = records['Close'] - records['ref']
        
        #because of weekends, there is often no reference availabel resulting in Nan
        # we can interpolate (simple linear) these missing momentums
        if interpolate_over_weekends:
            records['diff'].interpolate(method='linear', limit=2, inplace=True)
        
        
        #subset and format return
        retdf = records[['Date', 'diff']]
        retdf = (retdf.reset_index()
                 .set_index(pd.to_datetime(retdf['Date']))
                 .rename(columns={'diff': trg_series_name}))
        return retdf[[trg_series_name]]
        
      
    def get_RSI(self, historysize='1y', scale_period='14d', trg_series_name='rsi'):
        records = self.get_data(historysize)[self.analysis_stockvalue].to_frame()
        
        change = records["Close"].diff(periods=1) #previous day

        #split into positive market days and negative market days
        changeup = change.copy()
        changeup.loc[changeup<0] = 0
        changedown = change.copy()
        changedown.loc[changedown>0] = 0
        
        #rolling averages
        rolavg_downchange = changedown.rolling(window=scale_period).mean().abs() #make shure it is positive
        rolavg_upchange = changeup.rolling(window=scale_period).mean().abs()

        #compute index
        rsi = 100 * (rolavg_upchange / (rolavg_upchange + rolavg_downchange))
      
        rsi.name = trg_series_name
        return rsi.to_frame()
      
        
    def get_resistance_levels(self):
        pass
    
    
    