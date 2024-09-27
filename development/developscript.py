#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:37:02 2024

@author: thoverga
"""



import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ndaag 



zelf = ndaag.Stock('AAPL')

zelf.get_rolling_average().plot()
# ax = zelf.plot('1y')
# zelf.get_rolling_average().plot(ax=ax)

zelf.get_MACD_and_signal().plot()

# period='3mo'

# df = zelf.tick.history(period)    


# import matplotlib.gridspec as gridspec
# f = plt.figure()

# gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
              
# ax1 = plt.subplot(gs[0])
# ax2 = plt.subplot(gs[1], sharex=ax1)



# df['Close'].plot(ax=ax1)
# df['Volume'].plot(ax=ax2)


# plt.show()
# fig, axs = plt.subplots(2, 1, sharex=True, layout='constrained')













# appl = ndaag.Stock('AAPL')
# ret = appl.plot()

# appl.get_rolling_average()

# todays_data = ticker.history(period='1d')

# create ticker for Apple Stock
# ticker = yf.Ticker('AAPL')
# get data of the most recent date

# print(todays_data)



