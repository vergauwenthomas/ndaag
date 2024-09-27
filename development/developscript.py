#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:37:02 2024

@author: thoverga
"""



import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import ndaag 



zelf = ndaag.Stock('AAPL')

eventsdf = zelf.event_analysis()


# eventsdf['buy-trigger'] = eventsdf[['MACD_buy_signal',
#                                     'momentum_buy_trigger',
#                                     'rsi_buy_trigger']].apply(np.all, axis='columns')
# buy_triggers = eventsdf[eventsdf['buy-trigger']].index.to_list()





# print(buy_triggers)

# ax=zelf.plot(period='1y')

# (ymin, ymax) = ax.get_ylim()

# ax.vlines(x=buy_triggers, ymin=ymin, ymax=ymax, colors='purple', ls='--', lw=2, label='vline_multiple - full height')
# # for trig in buy_triggers:
#     # plt.
    
# plt.show()











#%%

# test = zelf.get_MACD_and_signal()





#%%


# history='1y'
# rsi_smoothing='2d'
# rsi_threshold = 40.
# event='buy'




# datadf = zelf.get_data(period=history)
# signaldf = pd.DataFrame()
# signaldf.index = datadf.index.copy()


# macdf = zelf.get_MACD_and_signal(historysize=history)
# if event =='buy':
#     #buy conditions: mac larger than signal and mac just crossed signal?
#     #cond1: macd greater then signal
#     macdf['cond1'] = (macdf['MACD'] >= macdf['MACD_signal'])
    
#     #cond2: change of regime (lines are crossing)
#     macdf['regime'] = macdf['cond1'] #True if macd over signal, false if macd under signal
#     macdf['cond2'] =  (macdf['regime'] != macdf['regime'].shift())   
    
#     #regimechange from false to True
   
#     macdf['MACD-buy-signal'] = macdf['cond1'] & macdf['cond2']
    

#     signaldf = signaldf.merge(macdf[['MACD_buy_trigger']],
#                               how='left',
#                               left_index=True,
#                               right_index=True)
# else:
#     sys.exit(f'{event} not implemented')


# under 40 and increasing



# appl = ndaag.Stock('AAPL')
# ret = appl.plot()

# appl.get_rolling_average()

# todays_data = ticker.history(period='1d')

# create ticker for Apple Stock
# ticker = yf.Ticker('AAPL')
# get data of the most recent date

# print(todays_data)



