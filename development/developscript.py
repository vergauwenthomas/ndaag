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



applestock = ndaag.Stock('AAPL')


applestock.fetch_data(period='9mo')

momentum = ndaag.MomentumIndex(stock=applestock, window='10d', interpolate_weekend=True)
momentum.calculate()


macd = ndaag.MACDIndex(stock=applestock)
macd.calculate()

cci = ndaag.CCIIndex(stock=applestock)
cci.calculate()
# momentum.plot()


#%% MACD



#%% MACD Index


#

#%% Comprehensive Technical Analysis Dashboard

# Calculate all signals
momentum_buy = momentum.find_buy_signal(threshold=0.0, require_rising=True)
momentum_sell = momentum.find_sell_signal(threshold=0.0, require_falling=True)

macd_buy = macd.find_buy_signal(early_warning_threshold=0.0)
macd_sell = macd.find_sell_signal(early_warning_threshold=0.0)

cci_buy = cci.find_buy_signal(threshold=-100)
cci_sell = cci.find_sell_signal(threshold=100)

# Create figure with multiple subplots sharing x-axis
fig = plt.figure(figsize=(16, 14))

# Define grid with different heights for subplots
gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 2, 2, 2], hspace=0.05)

# Create subplots
ax_price = fig.add_subplot(gs[0])
ax_volume = fig.add_subplot(gs[1], sharex=ax_price)
ax_cci = fig.add_subplot(gs[2], sharex=ax_price)
ax_momentum = fig.add_subplot(gs[3], sharex=ax_price)
ax_macd = fig.add_subplot(gs[4], sharex=ax_price)

# 1. Stock Price Plot
ax_price.plot(applestock.data.index, applestock.data['Close'], 
              label='Close Price', color='black', linewidth=2)
ax_price.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax_price.set_title('Apple Stock - Technical Analysis Dashboard', 
                    fontsize=16, fontweight='bold', pad=20)
ax_price.legend(loc='upper left')
ax_price.grid(True, alpha=0.3)
ax_price.tick_params(labelbottom=False)

# Mark buy/sell signals on price chart
if len(momentum_buy) > 0:
    ax_price.scatter(momentum_buy, applestock.data.loc[momentum_buy, 'Close'],
                     color='lightgreen', marker='^', s=80, alpha=0.6, 
                     edgecolors='green', linewidths=1.5, zorder=5)
if len(macd_buy) > 0:
    ax_price.scatter(macd_buy, applestock.data.loc[macd_buy, 'Close'],
                     color='cyan', marker='^', s=80, alpha=0.6,
                     edgecolors='blue', linewidths=1.5, zorder=5)
if len(cci_buy) > 0:
    ax_price.scatter(cci_buy, applestock.data.loc[cci_buy, 'Close'],
                     color='yellow', marker='^', s=80, alpha=0.6,
                     edgecolors='orange', linewidths=1.5, zorder=5)

# 2. Volume Histogram
ax_volume.bar(applestock.data.index, applestock.data['Volume'], 
              color='gray', alpha=0.5, width=0.8)
ax_volume.set_ylabel('Volume', fontsize=10, fontweight='bold')
ax_volume.ticklabel_format(style='plain', axis='y')
ax_volume.tick_params(axis='y', labelsize=8)
ax_volume.tick_params(labelbottom=False)
ax_volume.grid(True, alpha=0.3, axis='y')

# 3. CCI Plot (using built-in plot but we need to customize for shared axis)
cci_data = cci.cci
ax_cci.plot(cci_data.index, cci_data.values, 
            label=f'CCI ({cci.parameters["period"]})', 
            color='blue', linewidth=1.5)
ax_cci.axhline(y=100, color='red', linestyle='--', linewidth=1.5, 
               alpha=0.7, label='Overbought (+100)')
ax_cci.axhline(y=-100, color='green', linestyle='--', linewidth=1.5, 
               alpha=0.7, label='Oversold (-100)')
ax_cci.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax_cci.axhspan(100, 300, alpha=0.1, color='red')
ax_cci.axhspan(-100, -300, alpha=0.1, color='green')
if len(cci_buy) > 0:
    ax_cci.vlines(x=cci_buy, ymin=ax_cci.get_ylim()[0], ymax=ax_cci.get_ylim()[1],
                  colors='green', linestyle='-', alpha=0.5, linewidth=2)
if len(cci_sell) > 0:
    ax_cci.vlines(x=cci_sell, ymin=ax_cci.get_ylim()[0], ymax=ax_cci.get_ylim()[1],
                  colors='red', linestyle='-', alpha=0.5, linewidth=2)
ax_cci.set_ylabel('CCI', fontsize=11, fontweight='bold')
ax_cci.legend(loc='upper left', fontsize=9)
ax_cci.grid(True, alpha=0.3)
ax_cci.tick_params(labelbottom=False)

# 4. Momentum Plot
ax_momentum.plot(momentum.momentum.index, momentum.momentum.values,
                 label=f'Momentum ({momentum.parameters["window"]})',
                 color='purple', linewidth=1.5)
ax_momentum.axhline(y=0, color='black', linestyle='--', linewidth=1)
if len(momentum_buy) > 0:
    ax_momentum.vlines(x=momentum_buy, ymin=ax_momentum.get_ylim()[0], 
                       ymax=ax_momentum.get_ylim()[1],
                       colors='green', linestyle='-', alpha=0.5, linewidth=2)
if len(momentum_sell) > 0:
    ax_momentum.vlines(x=momentum_sell, ymin=ax_momentum.get_ylim()[0], 
                       ymax=ax_momentum.get_ylim()[1],
                       colors='red', linestyle='-', alpha=0.5, linewidth=2)
ax_momentum.set_ylabel('Momentum', fontsize=11, fontweight='bold')
ax_momentum.legend(loc='upper left', fontsize=9)
ax_momentum.grid(True, alpha=0.3)
ax_momentum.tick_params(labelbottom=False)

# 5. MACD Plot
ax_macd.plot(macd.macd_line.index, macd.macd_line.values, 
             label='MACD Line', color='blue', linewidth=1.5)
ax_macd.plot(macd.signal_line.index, macd.signal_line.values, 
             label='Signal Line', color='red', linewidth=1.5)
colors = ['green' if val >= 0 else 'red' for val in macd.histogram.values]
ax_macd.bar(macd.histogram.index, macd.histogram.values, 
            label='Histogram', color=colors, alpha=0.3, width=0.8)
ax_macd.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
if len(macd_buy) > 0:
    ax_macd.vlines(x=macd_buy, ymin=ax_macd.get_ylim()[0], 
                   ymax=ax_macd.get_ylim()[1],
                   colors='green', linestyle='-', alpha=0.5, linewidth=2)
if len(macd_sell) > 0:
    ax_macd.vlines(x=macd_sell, ymin=ax_macd.get_ylim()[0], 
                   ymax=ax_macd.get_ylim()[1],
                   colors='red', linestyle='-', alpha=0.5, linewidth=2)
ax_macd.set_ylabel('MACD', fontsize=11, fontweight='bold')
ax_macd.set_xlabel('Date', fontsize=12, fontweight='bold')
ax_macd.legend(loc='upper left', fontsize=9)
ax_macd.grid(True, alpha=0.3)

# Rotate x-axis labels for better readability
plt.setp(ax_macd.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add signal summary text
signal_summary = f"""
Signal Summary:
CCI: {len(cci_buy)} buys, {len(cci_sell)} sells
Momentum: {len(momentum_buy)} buys, {len(momentum_sell)} sells  
MACD: {len(macd_buy)} buys, {len(macd_sell)} sells
"""
fig.text(0.02, 0.98, signal_summary, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("TECHNICAL ANALYSIS SUMMARY")
print("="*60)
print(f"\nCCI Signals:")
print(f"  Buy signals:  {len(cci_buy)}")
print(f"  Sell signals: {len(cci_sell)}")
print(f"\nMomentum Signals:")
print(f"  Buy signals:  {len(momentum_buy)}")
print(f"  Sell signals: {len(momentum_sell)}")
print(f"\nMACD Signals:")
print(f"  Buy signals:  {len(macd_buy)}")
print(f"  Sell signals: {len(macd_sell)}")
print("="*60)









# idx = ndaag.MomentumIndex(stock=applestock, historysize='1y',
#                           window='10d', interpolate_weekend=True)




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



