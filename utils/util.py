""" Some utility code used in most files """

import os
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


""" Return CSV file path given a stock ticker symbol """
def symbol_to_path(symbol, folder='data'):
    return os.path.join(folder, '{}.csv'.format(symbol))


""" Reads adjusted close stock data for given symbols from CSV files """
def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',  
                usecols=['Date', 'Adj Close'], parse_dates=True, na_values=['NaN'] )
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        
        if symbol == 'SPY':
            df = df.dropna()
      
    return df


""" Plot stock prices with a custom title and axis labels """
def plot_data(df, title="Stock prices", xlabel="Data", ylabel="Prices"):
    ax = df.plot(title=title, fontsize=12, figsize=(15,8))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    plt.show()

    
""" Plot stock prices with bollinger bands, custom title, stock label and axis labels """
def plot_bollinger_band(df, window=20, title='Bollinger Bands', stock_label='Stock'):
    
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
    

""" Compute daily returns """    
def compute_daily_returns(df):
    daily_return = df.copy()
    daily_return[1:] = (df[1:] / df[:-1].values) - 1
    #daily_return = (df / df.shift(1)) - 1
    if daily_return.ndim == 1:
        daily_return.ix[0] = 0
    else:
        daily_return.ix[0,:] = 0
    return daily_return
