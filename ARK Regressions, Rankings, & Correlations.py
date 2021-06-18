#!/usr/bin/env python
# coding: utf-8

# ### Load Libraries

# In[1]:


# Load Packages
import statsmodels.api as sm # for regressions

import pandas as pd
import os
import io
import requests
import pandas_datareader as pdr
import matplotlib.pyplot as plt

import numpy as np


# ### Define Functions

# In[497]:


# Performs linear Regression
def lin_reg(x,y):
    """
    this function does linear regression
    Args:
        x (numpy.array, pandas.Series, pandas.DataFrame): independent variables
        y (numpy.array, pandas.Series, pandas.DataFrame): dependent variable
    """
    # Add a constant to the independent value
    x = sm.add_constant(x)
    # make regression model and fit results
    ols = sm.OLS(y, x).fit(cov_type='HC1', use_t = True)
    return ols


# ### Set Directory

# In[3]:


# Set Directory
data_dir = "G:\\Shared drives\\Monroe\\Lenovo\\NYU\\2021\\Linear Algebra\\Project\\Python\\"

os.chdir(data_dir)
print(os.listdir(), '\n', os.getcwd())

# Set start date
# start_date = '03/01/2016'
start_date = '10/29/2014'
end_date = '05/01/2021'


# ### Load Fama French Factors

# #### Load 3 Factors

# In[4]:


# Load Factors
ff3_factors = pd.read_csv('F-F_Research_Data_Factors_daily.csv', skiprows = 3, index_col = 0)

# Drop last column (copyright)
ff3_factors = ff3_factors[:-1]

# Index to Date
ff3_factors.index = pd.to_datetime(ff3_factors.index, format= '%Y%m%d')

# Rename Index
ff3_factors.index.name = 'Date'

# Convert to decimals (come as percents)
ff3_factors = ff3_factors.apply(lambda x: x/ 100)
ff3_factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)

print(ff3_factors.tail())


# #### Load 5 Factors

# In[691]:


#Load 5-factors
ff5_factors = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.CSV', skiprows = 3, index_col = 0)

# Index to Date
ff5_factors.index = pd.to_datetime(ff5_factors.index, format= '%Y%m%d')

# Rename Index
ff5_factors.index.name = 'Date'

# Convert to decimals (come as percents)
ff5_factors = ff5_factors.apply(lambda x: x/ 100)
ff5_factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)

print(ff5_factors.tail())


# #### Load Momentum Factor

# In[692]:


# Load Momentum
mom_factor = pd.read_csv('F-F_Momentum_Factor_daily.csv', skiprows=13, index_col=0)

# Drop last column (copyright)
mom_factor = mom_factor[:-1]

# Strip Spaces from column name
mom_factor.columns = mom_factor.columns.str.replace(' ', '')
mom_factor.rename(columns={'Mom': 'MOM'}, inplace=True)

# Index to Date
mom_factor.index = pd.to_datetime(mom_factor.index, format= '%Y%m%d')

# Rename Index
mom_factor.index.name = 'Date'

# Convert to decimals (come as percents)
carhart = mom_factor.apply(lambda x: x/ 100)

# Merge Momentum w/ 3 Factors
carhart = ff3_factors.merge(carhart, on='Date', left_index = True)

carhart


# #### Plot Factor Performance

# In[693]:


# Plot Factors over Observation Period
factor_plot = ff5_factors.merge(mom_factor.apply(lambda x: x/ 100), on='Date')

factor_plot[start_date:].cumsum().plot(title='Factor Performance')


# ### Load ARKK CSV file

# In[8]:


# Temporary loading of portfolio holdings. (Need to automate process of downloading holdings from ARK's website)

ARKK = pd.read_csv('PythonARKK_raw_data.csv')
# print(ARKK)

# Create list of tickers
ARKK_holdings = ARKK['ticker'].tolist()

# Remove space from tickers
ARKK_holdings = [str(w).replace('TREE UW', 'TREE').replace("ONVO ", 'ONVO') for w in ARKK_holdings]
print('\n', ARKK_holdings)


# ### Load  Yahoo Finance Data

# In[82]:


start_date = '10/29/2014'
# end_date = '04/23/2021'

# end_date = '05/01/2021'
end_date = '05/14/2021'

# Market Benchmarks
benchmark_list = ["^GSPC", "^DJI", "^IXIC", "XLK"] 

# ARK ETFs (Minus ARKX)
ARKtickers = ['ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF', 'ARKX']

# ARKK_holdings (decide whether to import basket of stocks)
# StockData = pdr.DataReader(ARKK_holdings, 'yahoo', start_date, end_date) 

ARK_funds = pdr.DataReader(ARKtickers, 'yahoo', start_date, end_date)
benchmark = pdr.DataReader(benchmark_list, 'yahoo', start_date, end_date)

# Rename Columns
benchmark = benchmark['Adj Close'].rename(columns = {'^GSPC':'sp500', '^DJI':'DowJones', '^IXIC':'NASDAQ'})


# ### S&P 500, Dow Jones NASDAQ

# In[436]:


# Market Benchmarks
LongTermbenchmark_list = ["^GSPC", "^DJI", "^IXIC"] 

LongTermbenchmark = pdr.DataReader(LongTermbenchmark_list, 'yahoo', '01/01/1970')
LongTermbenchmark = LongTermbenchmark['Adj Close'].rename(columns = {'^GSPC':'sp500', '^DJI':'DowJones', '^IXIC':'NASDAQ'})

# Calculate Lon-Term Benchmark Log Return
LongTermbenchmark_return = LongTermbenchmark.apply(np.log).diff(1).cumsum().apply(lambda x: x*100) 

LongTermbenchmark_return[:'12/31/2020'].plot(title='Long-Term Stock Returns 2000-2020')
LongTermbenchmark_return


# In[84]:


# Average Annualized Return for Dow Jones
print(219.523633/(2019-1990))
print(219.676356/(2019-1990))


# In[458]:


# Calculate Benchmark Log Return
benchmark_return = benchmark.apply(np.log).diff(1)

# # Calculate Cumulative Benchmark Return
# Cumulative_benchmark_return = benchmark_return.cumsum()

## Adjusted Returns, Download Risk-Free Rate of Return
benchmark_adj_return = ff5_factors[['RF']].merge(benchmark_return, on='Date', left_index = True)
benchmark_adj_return['sp500'] = benchmark_adj_return['sp500'] - benchmark_adj_return['RF']
benchmark_adj_return['DowJones'] = benchmark_adj_return['DowJones'] - benchmark_adj_return['RF']
benchmark_adj_return['NASDAQ'] = benchmark_adj_return['NASDAQ'] - benchmark_adj_return['RF']
benchmark_adj_return['XLK'] = benchmark_adj_return['XLK'] - benchmark_adj_return['RF']

# Plot Benchmark Returns
benchmark_adj_return.cumsum().drop(['RF','XLK'], axis = 1).apply(lambda x: x*100).plot(title = "Benchmark Daily Adjusted Returns")
plt.axvline(x='2020-03-23', color = 'red')
plt.text(x='2019-03-01', y=100, s='March 23')


# ### ARK ETFs

# In[86]:


## Merge ARK Funds
# Calculate ARK Log Return
ARK_return = ARK_funds['Adj Close'].apply(np.log).diff(1)

# # Calculate Cumulative ARK Return
# Cumulative_ARK_return = ARK_return.cumsum()

# Merge NASDAQ & XLK (Creating Unadjusted Returns)
ARK_UNadj_return = ARK_return.merge(benchmark_adj_return[['NASDAQ', 'XLK']], on='Date', left_index = True)

# Plot Non Ajdusted ARK Returns
ARK_UNadj_return.drop(['XLK', 'ARKX', 'ARKF'], axis= 1).cumsum().apply(lambda x: x*100).plot(title = 'ARK Unadjusted Returns')
plt.text(x='2019-03-01', y=200, s='March 23')
plt.axvline(x='2020-03-23', color = 'red')
plt.axhline(y=0, color='black', linestyle='dashed', linewidth = 1)


# #### Unadjusted Return

# In[87]:


# Plot Non Ajdusted ARK Returns w/ Unadjusted Benchmarks ***
ARK_UNadj_return.drop(['NASDAQ','XLK','ARKX', 'ARKF'], axis = 1).merge(benchmark_adj_return.drop(['XLK', 'RF'], axis = 1), on= 'Date').cumsum().apply(lambda x: x*100).plot(linewidth = 1, title="ARK ETF's Unadjusted Returns Relative to Benchmark Indices")
plt.text(x='2019-03-01', y=200, s='March 23')
plt.axvline(x='2020-03-23', color = 'red')
plt.axhline(y=0, color='black', linestyle='dashed', linewidth = 1)


# #### Adjusted Return (RF = Nasdaq)

# In[444]:


benchmark_return

# Plot Benchmark Returns
benchmark_return.cumsum().apply(lambda x: x*100).plot(title = "Benchmark Daily Returns")
plt.axvline(x='2020-03-23', color = 'red')
plt.text(x='2019-03-01', y=100, s='March 23')

# Merge Cummulative Returns, NASDAQ & XLK (Creating Adj. Returns) -----------------------------------
ARK_adj_return = ARK_return.merge(benchmark_return[['NASDAQ', 'XLK']], on='Date', left_index = True)

# Perform Adjustment Calculationr
ARK_adj_return['ARKK'] = ARK_adj_return['ARKK'] - ARK_adj_return['NASDAQ']
ARK_adj_return['ARKQ'] = ARK_adj_return['ARKQ'] - ARK_adj_return['NASDAQ']
ARK_adj_return['ARKW'] = ARK_adj_return['ARKW'] - ARK_adj_return['NASDAQ']
ARK_adj_return['ARKF'] = ARK_adj_return['ARKF'] - ARK_adj_return['NASDAQ']
ARK_adj_return['ARKX'] = ARK_adj_return['ARKX'] - ARK_adj_return['NASDAQ']

# Drop Excluded Stocks
ARK_adj_return = ARK_adj_return.drop(['XLK', 'ARKX', 'ARKF'], axis= 1)

# ARK + Unadjusted Benchmarks (Risk Adjusted)
#ARK_adj_return.drop('NASDAQ', axis = 1).merge(Cumulative_benchmark_return.drop(['XLK', 'NASDAQ'], axis = 1), on= 'Date').apply(lambda x: x*100).plot(linewidth = 1, title = "ARK ETF's Excess Returns")
ARK_adj_return.drop('NASDAQ', axis = 1).apply(lambda x: x*100).cumsum().plot(linewidth = 1, title = "ARK ETF's Excess Returns")
plt.text(x='2019-03-01', y=150, s='March 23')
plt.axvline(x='2020-03-23', color = 'red')
plt.axhline(y=0, color='black', linestyle='dashed', linewidth = 1)

# Plot Non Ajdusted ARK Returns w/ Unadjusted Benchmarks ***
ARK_UNadj_return.drop(['NASDAQ','XLK','ARKX'], axis = 1).merge(benchmark_return.drop(['XLK'], axis = 1), on= 'Date').apply(lambda x: x*100).cumsum().plot(linewidth = 1, title="ARK ETF's Unadjusted Returns Relative to Benchmark Indices")
plt.text(x='2019-03-01', y=200, s='March 23')
plt.axvline(x='2020-03-23', color = 'red')
plt.axhline(y=0, color='black', linestyle='dashed', linewidth = 1)


# # Presentation Figures

# In[431]:


# Compare ARK & Benchmark Return
benchmark_return
ARK_Comparison  = ARK_return.merge(benchmark_return, on='Date', left_index = True).drop(['XLK','ARKX'], axis = 1)
ARK_Comparison.apply(lambda x: x*100).cumsum().plot(linewidth = 1, title = "ARK vs Benchmark Returns")
plt.text(x='2019-03-01', y=150, s='March 23')
plt.axvline(x='2020-03-23', color = 'red')
plt.axhline(y=0, color='black', linestyle='dashed', linewidth = 1)

# Compare 2020
ARK_Comparison['2020-01-01':'2020-12-31'].apply(lambda x: x*100).cumsum().plot(linewidth = 1, title = "ARK vs Benchmark Returns 2020")
plt.text(x='2020-04-21', y=90, s='March 23')
plt.axvline(x='2020-03-23', color = 'red')
plt.axhline(y=0, color='black', linestyle='dashed', linewidth = 1)

# Compare 2020 (drop all benchmarks)
benchmark_drop = ['DowJones', 'sp500', 'NASDAQ']
ARK_Comparison['2020-01-01':'2020-12-31'].apply(lambda x: x*100).cumsum().drop(benchmark_drop, axis = 1).plot(linewidth = 1, title = "ARK Excess Returns 2020")
plt.text(x='2020-04-21', y=90, s='March 23')
plt.axvline(x='2020-03-23', color = 'red')
plt.axhline(y=0, color='black', linestyle='dashed', linewidth = 1)


# In[669]:


#2021 Benchmark Return
Bencmarks2021 = ARK_Comparison[['sp500', 'DowJones', 'NASDAQ']]['2021-01-01':].apply(lambda x: x*100).cumsum()

Bencmarks2021.plot(title = '2021 Benchmark Returns')
plt.axhline(y=0, color='black', linestyle='dashed', linewidth = 1)


Bencmarks2021


ARK_Comparison['2021-01-01':].apply(lambda x: x*100).cumsum().plot(title = 'ARK ETFs 2021 Performance')
plt.axhline(y=0, color='r', linestyle='-', linewidth = 3)
# plt.axhline(y=0, color='black', linestyle='dashed', linewidth = 1)


# In[718]:


#2021 Benchmark Return
Bencmarks2021 = ARK_Comparison[['sp500', 'DowJones', 'NASDAQ']]['2021-01-01':].apply(lambda x: x*100).cumsum()

Bencmarks2021.plot(title = 'Benchmark Returns 2021')
plt.axhline(y=0, color='black', linestyle='dashed', linewidth = 1)


Bencmarks2021


# In[667]:


# Merge Cummulative Returns, NASDAQ & XLK (Creating Adj. Returns) -----------------------------------
ARK_adj_return = ARK_return.merge(benchmark_return[['NASDAQ', 'XLK']], on='Date', left_index = True)

# Perform Adjustment Calculationr
ARK_adj_return['ARKK'] = ARK_adj_return['ARKK'] - ARK_adj_return['NASDAQ']
ARK_adj_return['ARKQ'] = ARK_adj_return['ARKQ'] - ARK_adj_return['NASDAQ']
ARK_adj_return['ARKW'] = ARK_adj_return['ARKW'] - ARK_adj_return['NASDAQ']
ARK_adj_return['ARKF'] = ARK_adj_return['ARKF'] - ARK_adj_return['NASDAQ']
ARK_adj_return['ARKX'] = ARK_adj_return['ARKX'] - ARK_adj_return['NASDAQ']


# Drop Excluded Stocks
ARK_adj_returnFig2 = ARK_adj_return['2021-01-01':].drop('XLK', axis = 1)

# ARK + Unadjusted Benchmarks (Risk Adjusted)
ARK_adj_returnFig2.drop('NASDAQ', axis = 1).apply(lambda x: x*100).cumsum().plot(linewidth = 1, title = "ARK Excess Returns 2021")
plt.axhline(y=0, color='black', linestyle='dashed', linewidth = 1)


# Return
ARK_adj_returnFig2.apply(lambda x: x*100).cumsum()


# In[16]:


ARK_UNadj_return[['ARKK','NASDAQ']].cumsum().apply(lambda x: x*100).plot(title = "ARKK & NASDAQ Daily Returns (Unadjusted)")


# In[17]:


ARK_funds = pdr.DataReader(ARKtickers, 'yahoo', start_date, end_date)


ARK_funds['2020-01-01':'2021-01-01'].apply(np.log).diff().apply(lambda x: x*100).cumsum()

# ARK_UNadj_return['2020-01-01':'2021-01-01'].cumsum().apply(lambda x: x*100)
# ARK_UNadj_return['2020-01-01':'2020-12-31'].drop(['NASDAQ','XLK','ARKX', 'ARKF'], axis = 1).apply(lambda x: x*100).cumsum()


# In[19]:


# ARK Cumulative Returns 2021
Return2021 = ARK_return['2021-02-01':]
Return2021.cumsum().apply(lambda x: x*100).plot(title='ARK Cumulative Returns 2021')
plt.axhline(y=0, color='black', linestyle='dashed', linewidth = 1)


# In[20]:


# ARKK vs. Benchmark Returns
ARK_benchmark = pdr.DataReader(benchmark_list + ["ARKK"], 'yahoo', start_date, end_date)
ARK_benchmark = ARK_benchmark['Adj Close'].rename(columns = {'^GSPC':'sp500', '^DJI':'DowJones', '^IXIC':'NASDAQ'})

# Calculate Benchmark Log Return
ARK_benchmark_return = ARK_benchmark.apply(np.log).diff(1)

# Calculate Cumulative Benchmark Return
Cumulative_ARK_benchmark_return = ARK_benchmark_return.cumsum().apply(np.exp)


# ## VOO & SP500 Comparison

# In[671]:


# VOO vs. SP 500
VOO_benchmark = pdr.DataReader(["^GSPC", "VOO"], 'yahoo', start_date, end_date)
VOO_benchmark = VOO_benchmark['Adj Close'].rename(columns = {'^GSPC':'sp500'})

# Calculate Benchmark Log Return
VOO_benchmark_return = VOO_benchmark.apply(np.log).diff(1)

# Calculate Cumulative Benchmark Return
Cumulative_VOO_benchmark_return = VOO_benchmark_return.cumsum()

# Plot Benchmark Return
# Cumulative_VOO_benchmark_return.apply(lambda x: x*100).plot(title='Voo Returns vs Benchmark daily returns')

## Estimate Risk-Adjusted Return
VOOExcessReturn = ff5_factors[['RF']].merge(VOO_benchmark_return, on='Date', left_index = True)
VOOExcessReturn['VOO'] = VOOExcessReturn['VOO'] - VOOExcessReturn['RF']
VOOExcessReturn['sp500'] = VOOExcessReturn['sp500'] - VOOExcessReturn['RF']
VOOExcessReturn = VOOExcessReturn.drop('RF', axis = 1)

# Remove NA Values
VOOExcessReturn = VOOExcessReturn.dropna()

# Calculate cumulative Excess Return
Cumulative_VOOExcessReturn = VOOExcessReturn.cumsum()

# Plot Excess Return
Cumulative_VOOExcessReturn.apply(lambda x: x*100).plot(title='VOO & SP500 Daily Returns')
plt.text(x='2019-03-01', y=60, s='March 23')
plt.axvline(x='2020-03-23', color = 'red')
plt.axhline(y=0, color='black', linestyle='dashed', linewidth = 1)


# In[672]:


# CAPM VOO Excess Return
X = VOOExcessReturn[['sp500']]
y = VOOExcessReturn[['VOO']]

ols = lin_reg(X, y)
ols.summary()


# ### Plotting Risk-Adjusted Returns (Excess Returns)

# In[21]:


ln_ff5 = ARK_adj_return.merge(factor_plot, on='Date', left_index = True).cumsum().dropna()

# Check Plots
ln_ff5.plot()
ARK_adj_return.cumsum().plot()
factor_plot[start_date:].cumsum().plot()

# Long-Term Factor Plot
factor_plot.cumsum().plot()


# ## CAPM, 3-Factor, 5-Factor, 4-Factor, 5-Factor + Momentum

# ### CAPM

# In[708]:


# Regression CAPM Model
X = ln_ff5[['MKT']]
y = ln_ff5['ARKK']

ols = lin_reg(X, y)
ols.summary()


# ### 3-Factor

# In[674]:


# Regression 3-Factor Model
X = ln_ff5[['MKT','SMB','HML']]
y = ln_ff5['ARKK']

ols = lin_reg(X, y)
ols.summary()


# ### 4-Factor

# In[675]:


# Regression Carhart 4-Factor Model
X = ln_ff5[['MKT','SMB','HML', 'MOM']]
y = ln_ff5['ARKK']

ols = lin_reg(X, y)
ols.summary()


# ### 5-factor

# In[694]:


# Regression 5-Factor Model
X = ln_ff5[['MKT','SMB','HML', 'RMW', 'CMA']]
y = ln_ff5['ARKK']

ols = lin_reg(X, y)
ols.summary()


# In[710]:


# Regression 5-Factor Model + Momentum
X = ln_ff5[['MKT','SMB','HML', 'RMW', 'CMA', "MOM"]]
y = ln_ff5['ARKK']

ols = lin_reg(X, y)
ols.summary()


# ### 2020

# In[709]:


# Regression CAPM Model 2020
X = ln_ff5['2020-01-01':'2020-12-31'][['MKT']]
y = ln_ff5['2020-01-01':'2020-12-31']['ARKK']

ols = lin_reg(X, y)
ols.summary()


# In[702]:


# Regression 3-Factor Model 2020
X = ln_ff5['2020-01-01':'2020-12-31'][['MKT','SMB','HML']]
y = ln_ff5['2020-01-01':'2020-12-31']['ARKK']

ols = lin_reg(X, y)
ols.summary()


# In[703]:


# Regression 5-Factor Model 2020
X = ln_ff5['2020-01-01':'2020-12-31'][['MKT','SMB','HML', 'RMW', 'CMA']]
y = ln_ff5['2020-01-01':'2020-12-31']['ARKK']

ols = lin_reg(X, y)
ols.summary()


# In[704]:


# Regression 4-Factor Model 2020
X = ln_ff5['2020-01-01':'2020-12-31'][['MKT','SMB','HML', 'MOM']]
y = ln_ff5['2020-01-01':'2020-12-31']['ARKK']

ols = lin_reg(X, y)
ols.summary()


# ### 6-Factor

# ## ARKK Holdings Portfolio Regression

# In[512]:


# ARKK_holdings (decide whether to import basket of stocks)
StockData = pdr.DataReader(ARKK_holdings + ['ARKK', 'ARKW', 'ARKQ'], 'yahoo', start_date, end_date) 


# In[575]:


Holdings = StockData['Adj Close']

# Caclulate Log Returns
HoldingsReturn = Holdings['2020-01-01':][1:].drop(['ARKW', 'ARKQ'], axis=1).apply(np.log).diff().dropna()

print(len(HoldingsReturn.columns))

#print(Cumulative_Holdings[['TSLA', 'TDOC']].to_string())


# In[578]:


X = HoldingsReturn.drop('ARKK', axis=1).cumsum()
y = HoldingsReturn[['ARKK']].cumsum()

ols = lin_reg(X, y)
ols.summary()


# In[24]:


X = HoldingsReturn.drop('ARKK', axis=1).cumsum()
y = HoldingsReturn[['ARKK']].cumsum()

ols = lin_reg(X, y)
ols.summary()


# #### Return Significant Coeficients (p<0.001)

# In[579]:


# Here we select p-values that are significant

# Pull tickers with p-values from ols regression
t = dict(ols.pvalues[:])

# Reformat data fame
t = pd.DataFrame(t.items(), columns=['Symbols', 'p-value'])

# Make p-value type to numeric
t['p-value'] = pd.to_numeric(t['p-value'])

# Rank p-values from smalles to largest
t.loc[:,'rank'] = t['p-value'].rank(method='dense', ascending=True)

# Reorder tikcers by rank
t = t.sort_values('rank').reset_index(drop=True)

# Keep only ticker that are statistically signficant (p<0.1)
Significant = t[t['p-value'] <= .001]

print(Significant['Symbols'].tolist())
Significant


# In[580]:


#Sum Coefficients

# Pull tickers with p-values from ols regression
t = dict(ols.params[:])

# Reformat data fame
t2 = pd.DataFrame(t.items(), columns=['Symbols', 'Coefficient'])

test = Significant.merge(t2)

# Sum Coefficients
print(sum(test['Coefficient']))
print(sum(ols.params))

# test
test[test['rank'] < 6]


# ## Tesla

# In[581]:


TSLA_ARKK = Holdings[['TSLA','ARKK']]['2020-03-23':].apply(np.log).diff().cumsum().dropna()
#TSLA_ARKK=TSLA_ARKK.apply(lambda x: x*100)


# In[583]:


X = TSLA_ARKK['TSLA']
y = TSLA_ARKK['ARKK']

ols = lin_reg(X, y)
ols.summary()


# In[584]:


plt.scatter(TSLA_ARKK['TSLA'], TSLA_ARKK['ARKK'])
plt.xlabel("TSLA")
plt.ylabel("ARKK")
plt.title("Tesla (TSLA) and ARK Innovation ETF (ARKK) Returns")


# In[502]:


SQ_ARKK = Holdings[['SQ','ARKK']]['2020-03-23':].apply(np.log).diff().cumsum().dropna()

X = SQ_ARKK ['SQ']
y = SQ_ARKK ['ARKK']

ols = lin_reg(X, y)
ols.summary()


# In[503]:


SE_ARKK = Holdings[['SE','ARKK']]['2020-03-23':].apply(np.log).diff().cumsum().dropna()

X = SE_ARKK ['SE']
y = SE_ARKK ['ARKK']

ols = lin_reg(X, y)
ols.summary()


# In[506]:


Holdings[['TDOC']]['2020-03-23':].apply(np.log).diff().cumsum()


# ## Find and Plot Returns of Top 10 & Bottom 10 stocks
# 

# In[30]:



# Merge Cummulative Returns, NASDAQ & XLK (Creating Adj. Returns) -----------------------------------
ARK_adj_return = ARK_return.merge(benchmark_adj_return[['NASDAQ', 'XLK']], on='Date', left_index = True)

table = ARK_adj_return['2020-01-01':'2021-01-01'].cumsum().apply(lambda x: x*100)


# Perform Adjustment Calculationr
ARK_adj_return['ARKK'] = ARK_adj_return['ARKK'] - ARK_adj_return['NASDAQ']
ARK_adj_return['ARKQ'] = ARK_adj_return['ARKQ'] - ARK_adj_return['NASDAQ']
ARK_adj_return['ARKW'] = ARK_adj_return['ARKW'] - ARK_adj_return['NASDAQ']
ARK_adj_return['ARKF'] = ARK_adj_return['ARKF'] - ARK_adj_return['NASDAQ']
ARK_adj_return['ARKX'] = ARK_adj_return['ARKX'] - ARK_adj_return['NASDAQ']

table.plot


# In[31]:


HoldingsReturn = Holdings['2020-03-23':'2021-05-01'].drop(['ARKW', 'ARKQ', 'ARKK'], axis=1).apply(np.log).diff().cumsum()

Ranking_Holdings = HoldingsReturn.apply(lambda x: x*100)
Ranking_Holdings


# ### Top 10

# In[585]:


# Best Performers

# Keep ARK holdings & Indices
#Ranking_Holdings.drop(ARKtickers, 1)

R = Ranking_Holdings.reset_index(inplace = False)

#get most Resent Date
recent_date = R['Date'].max()

# Select max date row
maxRow = R[R['Date']==recent_date].melt(id_vars = ['Date'])
maxRow['rank'] = maxRow['value'].rank(ascending=False)
maxRow = maxRow[maxRow['value'].notna()].sort_values('rank', ascending = True).set_index('Date')

# Keep only Top 10
TopRank = maxRow[maxRow['rank'] <= 10]

print(TopRank['Symbols'].tolist())

print(TopRank)


# ### Bottom 10

# In[587]:


# Worst Performers

# Select Max date row
minRow = R[R['Date']==recent_date].melt(id_vars = ['Date'])
minRow['rank'] = minRow['value'].rank(ascending=True)
minRow = minRow[minRow['value'].notna()].sort_values('rank', ascending = True).set_index('Date')

#Keep only Bottom 10
BottomRank = minRow[minRow['rank'] <= 10]

print(BottomRank['Symbols'].tolist())

print(BottomRank)


# ### Full Ranking

# In[35]:


# Full Rankings
maxRow


# In[36]:


# Plotting Function
def my_plot(y, x=None, title='', x_label='', y_label='', fig_size=(8,6), marker='-'):
    """
    this function creates line plots 
    Args:
        y (numpy.array, pandas.Series, pandas.DataFrame): y-axis
        x (numpy.array, pandas.Series, pandas.DataFrame): x-axis
        title (string): plot title
        x_label (string): x-axis title
        y_label (string): y-axis title
        legend ('upper left', 'upper right', 'lower left', 'lower right',...): legend location
        marker = (‘solid’ | ‘dashed’, ‘dashdot’, ‘dotted’ | (offset, on-off-dash-seq) | '-' | '--' | '-.' | ':' | 'None' | ' ' | ''): linetype
    """
    legend = list(y.columns)
    f = plt.figure(figsize=fig_size)
    if x is None:
        plt.plot(y, linestyle=marker)
    else:
        plt.plot(x, y, linestyle=marker)
    

    plt.legend(legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    


# In[37]:


my_plot(Ranking_Holdings[TopRank['Symbols']], title = 'ARKK Top 10 Stocks')


# In[38]:


my_plot(Ranking_Holdings[BottomRank['Symbols']], title = 'Figure 2. Bottom 10 Stocks')
plt.axhline(y=0, color='r', linestyle='-', linewidth = 3)


# ### Plot Top 10 & Bottom 10 at a monthly frequency

# In[752]:


month_top10 = Ranking_Holdings[TopRank['Symbols']].resample('1M').mean()

my_plot(month_top10, title = '2020 Performance of Top 10')
plt.axhline(y=0, color='r', linestyle='-', linewidth = 3)


# In[753]:


month_bot10 = Ranking_Holdings[BottomRank['Symbols']].resample('1M').mean()

my_plot(month_bot10, title = '2020 Performance of Bottom 10')
plt.axhline(y=0, color='r', linestyle='-', linewidth = 3)


# # Correlation Matrices

# In[39]:


import seaborn as sns
import numpy.linalg as la

# Calculate Correaltion Matrices for returns
Topcorr = Ranking_Holdings[TopRank['Symbols']].corr()
Bottomcorr = Ranking_Holdings[BottomRank['Symbols']].corr()


# In[40]:


# Topcorr
# HoldingsReturn
cormatrix = HoldingsReturn.corr()

t = HoldingsReturn


la.det(cormatrix)
################ copy


# In[589]:


# Top 10 correlation
plt.figure(figsize=(20, 20))
sns.heatmap(Topcorr, annot = True, annot_kws={"size": 20}, cbar = False,)
sns.set(font_scale=4)
plt.xlabel("") 
plt.ylabel("")
# ax.set(xlabel="Ticker", ylabel = "Ticker")


# In[590]:


# Bottom 10 Correlation Matrix
plt.figure(figsize=(20, 20))
sns.heatmap(Bottomcorr, annot = True, annot_kws={"size": 20}, cbar = False)
sns.set(font_scale=4)
plt.xlabel("") 
plt.ylabel("")


# ### 2021 YTD Peformance of top 10 and bottom 10 2020

# In[52]:


plt.style.use('ggplot')


# In[592]:


plt.rcParams.update(plt.rcParamsDefault)


# In[748]:


# HoldingsReturn[['TDOC', 'Z']]['2021-01-01':].cumsum().plot()

HoldingsReturn2 = Holdings['2021-01-01':].drop(['ARKW', 'ARKQ', 'ARKK'], axis=1).apply(np.log).diff().cumsum().apply(lambda x: x*100)

HoldingsReturn2[TopRank['Symbols']]['2021-01-01':].resample('2W').mean().plot(title="2021 Performance of Top 10")
# plt.style.use('default')
plt.axhline(y=0, color='r', linestyle='-', linewidth = 3)
plt.legend(loc="upper left")
# plt.legend(loc="upper left")  #ncol=len(HoldingsReturn.colum.resample('1M').mean()ns))
# plt.figure(figsize=(15, 15))
# plt.tight_layout()




HoldingsReturn2[BottomRank['Symbols']]['2021-01-01':].resample('2W').mean().plot(title="2021 Performance of Bottom 10")
plt.axhline(y=0, color='r', linestyle='-', linewidth = 3)
plt.legend(loc="upper left")


#print(HoldingsReturn[TopRank['Symbols']]['2021-05-14':])
print(HoldingsReturn[BottomRank['Symbols']])


# In[656]:


Holdings


# In[736]:


HoldingsTop5[['TDOC']]['2021-01-01':]


# In[666]:



HoldingsTop5 = Holdings[['TSLA', 'TDOC', 'SQ', 'SHOP', 'ROKU']].apply(np.log).diff().apply(lambda x: x*100)

# Tesla 2020
HoldingsTop5[['TSLA']]['2020-01-01':'2020-12-31'].cumsum().plot(title = "Tesla Performance 2020")
plt.axhline(y=0, color='r', linestyle='-', linewidth = 3)

# Tesla 2021
HoldingsTop5[['TSLA']]['2021-01-01':].cumsum().plot(title = "Tesla 2021 Performance")
plt.axhline(y=0, color='r', linestyle='-', linewidth = 3)

# Teledoc 2020
HoldingsTop5[['TDOC']]['2020-01-01':'2020-12-31'].cumsum().plot(title = "Teledoc Performance 2020")
plt.axhline(y=0, color='r', linestyle='-', linewidth = 3)

# Teledoc 2021
HoldingsTop5[['TDOC']]['2021-01-01':].cumsum().plot(title = "Teledoc 2021 Performance")
plt.axhline(y=0, color='r', linestyle='-', linewidth = 3)




# Top 5 Holdings 2020
HoldingsTop5['2020-01-01':'2020-12-31'].cumsum().plot(title = "Top 5 Holdings by Market Value Performance 2020")
plt.axhline(y=0, color='r', linestyle='-', linewidth = 3)

# Top 5 Holdings 2021
HoldingsTop5['2021-01-01':].cumsum().plot(title = "Top 5 Holdings by Market Value Performance 2021")
plt.axhline(y=0, color='r', linestyle='-', linewidth = 3)


# # PCA

# In[56]:


# Factor = X*v/sqrt(lambda)
# Standardize X
# Standardize ARKK returns
# X.corr()

# X is returns


# In[57]:


def standardize(returns):
    
    return (returns - returns.mean()) /returns.std()


# In[ ]:


pdr.DataReader('ARKK', 'yahoo', '2020-03-23') .dropna()


# In[405]:


# ARKK_holdings (decide whether to import basket of stocks)
StockData = pdr.DataReader(ARKK_holdings + ['ARKK'], 'yahoo', '2020-03-23', end_date) 


# In[380]:


Holdings = StockData['Adj Close'].apply(np.log).diff().dropna()

Stockdata = Holdings[1:].drop(['ARKK'], axis=1)
ARKKdata = Holdings[['ARKK']][1:]


# In[381]:


import math

# Correlation matrix of standardized returns
X = standardize(Stockdata)
correlation_matrix = X.corr()

# Grab Larges E-values &  E-vector
evals, evecs = la.eig(correlation_matrix)

X = (Stockdata * evecs[:, np.argmax(evals)]) / np.sqrt(evals[0])
print(len(X))

y =  standardize(ARKKdata)
print(len(y))

ols = lin_reg(X, y)
ols.summary()


# In[382]:


X


# In[383]:


predicted_returns = pd.DataFrame(data = ols.predict(), index = ARKKdata.index)

plt.plot(predicted_returns.cumsum())
plt.plot(standardize(ARKKdata).cumsum())


# In[373]:


Holdings[1:].dropna(axis=0)


# In[410]:


drop_Data


# In[754]:


## Omit NA Version
# drop_Data = StockData['Adj Close']['2021-01-01':].apply(np.log).diff().dropna()
# drop_Data

# Stock_data = drop_Data.drop(['ARKK'], axis=1)
# ARKKdata = drop_Data[['ARKK']]

# print(len(StockData), len(ARKKdata))


# X = (Stock_data * evecs[:, np.argmax(evals)]) / np.sqrt(evals[0])
# print(len(X))

# y =  standardize(ARKKdata)
# print(len(y))

# # Add a constant to the independent value
# x = sm.add_constant(X)
# # make regression model and fit results
# ols = sm.OLS(y,x,missing='drop').fit()

# print(ols.summary())


# In[755]:


# predicted_returns = pd.DataFrame(data = ols.predict(), index = ARKKdata.index)

# plt.plot(predicted_returns.cumsum())
# plt.plot(standardize(ARKKdata).cumsum())


# In[ ]:


import math

# Correlation matrix of standardized returns
X = standardize(StockData)
correlation_matrix = X.corr()

# Grab Larges E-values &  E-vector
evals, evecs = la.eig(correlation_matrix)

print(evals[0])
print('----------------------')
print(evecs[:, np.argmax(evals)])
print('----------------------')
print(evecs)
print('----------------------')
print(evals)


# In[ ]:


evecs[:, np.argmax(evals)]


# In[ ]:


predicted_returns = pd.DataFrame(data = ols.predict(), index = ARKKdata.index)

plt.plot(predicted_returns.cumsum())
plt.plot(standardize(ARKKdata).cumsum())


# In[ ]:


plt.plot(ols.predict().cumsum())

plt.plot(standardize(ARKKdata).cumsum())


# In[242]:


plt.plot(standardize(ARKKdata).cumsum())


# In[214]:


# Here we select p-values that are significant

# Pull tickers with p-values from ols regression
t = dict(ols.pvalues[:])

# Reformat data fame
t = pd.DataFrame(t.items(), columns=['Symbols', 'p-value'])

# Make p-value type to numeric
t['p-value'] = pd.to_numeric(t['p-value'])

# Rank p-values from smalles to largest
t.loc[:,'rank'] = t['p-value'].rank(method='dense', ascending=True)

# Reorder tikcers by rank
t = t.sort_values('rank').reset_index(drop=True)

# Keep only ticker that are statistically signficant (p<0.1)
Significant = t[t['p-value'] <= .001]

print(Significant['Symbols'].tolist())
Significant

