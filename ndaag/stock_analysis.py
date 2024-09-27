#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:16:13 2024

@author: thoverga
"""


class StockdAnalysis:
    


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
