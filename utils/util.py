""" Some utility code used in most files """

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic(u'matplotlib inline')


def symbol_to_path(symbol, folder='data'):
    """ Return CSV file path given a stock ticker symbol """
    return os.path.join(folder, '{}.csv'.format(symbol))


def get_data(symbols, dates, price='Adj Close', folder='data', addSPY=True):
    """ Reads price stock data (Adj Close is the default) for given symbols from CSV files """
    df = pd.DataFrame(index=dates)
    
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols
    
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol, folder), index_col='Date',  
                usecols=['Date', price], parse_dates=True, na_values=['NaN'] )
        df_temp = df_temp.rename(columns={price: symbol})
        df = df.join(df_temp)
        
        if symbol == 'SPY':
            df = df.dropna()
      
    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Prices"):
    """ Plot stock prices with a custom title and axis labels """
    ax = df.plot(title=title, fontsize=12, figsize=(15,8))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    plt.show()

    
def plot_bollinger_band(df, window=20, title='Bollinger Bands', stock_label='Stock'):
    """ Plot stock prices with bollinger bands, custom title, stock label and axis labels """    
    def get_bollinger_bands(rm_Stock, rstd_Stock):
        upper_band = rm_Stock + 2*rstd_Stock
        lower_band = rm_Stock - 2*rstd_Stock
        return upper_band, lower_band

    # 1.Compute rolling mean
    rm_Stock = df.rolling(window=window).mean()
    # 2.Compute rolling standard deviation
    rstd_Stock = df.rolling(window=window).std()
    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm_Stock, rstd_Stock) 

    # Plot
    ax = df.plot(title=title, label=stock_label, figsize=(15,8))
    rm_Stock.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)

    # Add axis labels and legends
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid()
    plt.show()    
    

def compute_daily_returns(df):
    """ Compute daily returns """   
    daily_return = df.copy()
    daily_return[1:] = (df[1:] / df[:-1].values) - 1
    #daily_return = (df / df.shift(1)) - 1
    if daily_return.ndim == 1:
        daily_return.ix[0] = 0
    else:
        daily_return.ix[0,:] = 0
    return daily_return


def compute_cumulative_returns(df):
    """ Compute cumulative returns """   
    cum_ret = (df[-1] / df[0]) - 1
    return cum_ret


def compute_sharpe_ratio(df, daily_rf=0, samples_per_year=252):
    """ Compute sharpe ratio """   
    daily_returns = compute_daily_returns(df)
    sharpe_ratio = ((daily_returns - daily_rf).mean()/daily_returns.std()) * np.sqrt(samples_per_year)
    return sharpe_ratio


def get_portfolio_value(prices, allocs, start_val=1):
    """Computes the daily portfolio value given daily prices for each stock in portfolio, 
    initial allocations(as fractions that sum to 1) and total starting value invested in portfolio"""
    # normalize all stock prices
    df = prices / prices.ix[0]
    # multiply prices by allocations of each equity
    df = df * allocs
    # multiply allocated values by initial investment value
    df = df * start_val
    # compute entire portfolio value on each day
    port_val = df.sum(axis=1)
    return port_val


def plot_histogram(df, title='Histogram', xlabel='x-label'):
    """ Plot histogram and compute following statistics: mean, sdev and kurtosis """
    print 'mean=', df.mean()
    print 'st.dev=', df.std()
    print 'kurtosis=', df.kurtosis()
    ax = df.hist(bins=20, figsize=(14,8))
    # Add axis labels and other features
    plt.axvline(df.mean(), color='w', linestyle='dashed', linewidth=2)
    plt.axvline(-df.std(), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(df.std(), color='r', linestyle='dashed', linewidth=2)
    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('frequency')
    plt.show()
    
       
def fit_scatter(df, x, y):
    ''' Plot scatter and linearly fit the data '''
    df.plot(kind='scatter', x=x, y=y)
    plt.grid()
    beta_Stock, alpha_Stock = np.polyfit(df[x], df[y],1)
    print 'beta=', beta_Stock
    print 'alpha=', alpha_Stock
    plt.plot(df['SPY'], beta_Stock*df['SPY']+alpha_Stock, '-', color='r')
    plt.show()
    
    
def ticks_in_cluster(x, y, g_ell_center, g_ell_width, g_ell_height, angle):
    ''' Returns an array of colors: elements in the cluster are those marked green '''    
    cos_angle = np.cos(np.radians(180.-angle))
    sin_angle = np.sin(np.radians(180.-angle))

    xc = x - g_ell_center[0]
    yc = y - g_ell_center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle 

    rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)

    colors_array = []

    for r in rad_cc:
        if r <= 1.:
            # point in ellipse
            colors_array.append('green')
        else:
            # point not in ellipse
            colors_array.append('blue')
            
    return colors_array
