import pandas as pd
import numpy as np

# note trail_size includes the current record. 
def compute_highest_high(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the hioghest high. """
    return series['High'].rolling(window=trail_size, min_periods=1).max()


def compute_lowest_low(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the lowest low. """
    return series['Low'].rolling(window=trail_size, min_periods=1).min()


def compute_avg_vol(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the average volume. """
    return series['Volume'].rolling(window=trail_size, min_periods=1).mean()


def compute_sma(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the simple moving average. """
    return series['Close'].rolling(window=trail_size, min_periods=1).mean()


def compute_sd(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the standard deviation. """
    res = series['Close'].rolling(window=trail_size, min_periods=1).std()
    return res.replace(0, 0.1)


def compute_willr(series):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series Williams %R marker. """
    return -100*((series['High']-series['Close'])/(series['High']-series['Low']))


def compute_atr(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series Average True Range. """
    
    # get a series of the previous day's closes
    prev_closes = series['Close'].shift(1).fillna(method='bfill')
    
    # make a df of the 3 differences we want to max
    triple_diffs = pd.DataFrame()
    triple_diffs['highlow'] = series['High'] - series['Low']
    triple_diffs['highclose'] = abs(series['High'] - prev_closes)
    triple_diffs['closelow'] = abs(prev_closes - series['Low'])
    
    true_ranges = triple_diffs.max(axis=1)
    
    return true_ranges.rolling(window=trail_size, min_periods=1).mean()


def compute_dmh(series):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series Directional Movement High. """
    return series['High'].diff(1).fillna(0)


def compute_dml(series):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series Directional Movement Low. """
    return series['Low'].diff(1).fillna(0)


def compute_ema(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the exponential moving average. """
    
    return series['Close'].ewm(span=trail_size).mean()


def compute_wma(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the weighted moving average. """
    
    weights = np.arange(1, trail_size+1)
    weights = weights[::-1]
    #should produce array [trailsize+1, trailsize, ..., 2, 1]
    
    #SUM ((close[t]*N) + (close[t-1] *(N-1)) + ... + (close[t-N])) 
    #wma = wma/(sum of the weights)
    wma = series['Close'].rolling(window=trail_size).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    
    return wma


def compute_bbhigh(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the High Bollinger Bands. """
    #SMA[N] + SD[N]*2

    return series['SMA'] + series['SD']*2


def compute_bblow(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the Low Bollinger Bands. """
    #SMA[N] - SD[N]*2
    
    return series['SMA'] - series['SD']*2


def compute_perbhigh(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the Higher Limit Bollinger Bands. """
    #BBHIGH[N]/SD[N]
    return series['BBHIGH'].div(series['SD'])


def compute_perblow(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the Lower Limit Bollinger Bands. """
    #BBLOW[N]/SD[N]
    return series['BBLOW'].div(series['SD'])


def compute_trima(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the triangular moving average. """
    #(SMA[t-N] + ... + SMA[t])/N
    return (series['SMA'].rolling(window=trail_size).sum() / trail_size)


def compute_rsi(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the relative strength index. 
    This code was taken off stackoverflow. It produces warnings but
    does what we need it to do. Can't seem to figure out why it returns
    series rsi and RSI in the table"""
    new_series = series.copy().reset_index(drop=True)
    change = pd.DataFrame()
    change = new_series['Close'].diff(1) # Calculate change

    # calculate gain / loss from every change
    new_series['Gain'] = np.select([change>0, change.isna()], 
                        [change, np.nan], 
                        default=0) 
    new_series['Loss'] = np.select([change<0, change.isna()], 
                        [-change, np.nan], 
                        default=0)

    # create avg_gain /  avg_loss columns with all nan
    new_series['Avg_gain'] = np.nan 
    new_series['Avg_loss'] = np.nan


    # keep first occurrence of rolling mean
    new_series['Avg_gain'][trail_size] = new_series['Gain'].rolling(window=trail_size).mean().dropna().iloc[0] 
    new_series['Avg_loss'][trail_size] = new_series['Loss'].rolling(window=trail_size).mean().dropna().iloc[0]
    
    #Looping through the pandas series
    for i in range(trail_size+1, series.shape[0]):
        new_series['Avg_gain'].iloc[i] = (new_series['Avg_gain'].iloc[i-1] * (trail_size - 1) + new_series['Gain'].iloc[i]) / trail_size
        new_series['Avg_loss'].iloc[i] = (new_series['Avg_loss'].iloc[i-1] * (trail_size - 1) + new_series['Loss'].iloc[i]) / trail_size

    # calculate rs and rsi
    new_series['rs'] = new_series['Avg_gain'] / new_series['Avg_loss']
    rsi = 100 - (100 / (1 + new_series['rs'] ))

    return rsi.values


def compute_dx(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the directional index. """
    #(abs(+DM - -DM)/(+DM + -DM))*100 
    
    num = abs(series['DMH'] - series['DML'])
    den = abs(series['DMH'] + series['DML'])

    res = num / den * 100
    return res.replace([np.inf, -np.inf], 0)


def compute_positive_di(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the positive directional indicator. """
    #(+DM/ATR[N])*100 
    
    return (series['DMH'].div(series['ATR']).mul(100))


def compute_negative_di(series, trail_size=20):
    """A function to take in a dataframe for one stock time series, 
    and return a pandas series for the negative directional indicator. """
    #(-DM/ATR[N])*100 
    
    return (series['DML'].div(series['ATR']).mul(100))


def compute_roc(series,trail_size = 20):
    """A function to take in a dataframe for one stock time series,
    and return a panda series Rate of Change."""
    return (series['Close']/(series['Close'].shift(trail_size).fillna(method='bfill'))-1)*100


def compute_macd(series,trail_size = 20):
    """A function to take in a dataframe for one stock time series,
    and return a panda series Moving Average Convergance Divergence."""
    
    return compute_ema(series, trail_size) - compute_ema(series, 2*trail_size)


def compute_cci(series,trail_size = 20):
    """A function to take in a dataframe for one stock time series,
    and return a panda series Commonodity Channel Index."""
    
    num = (series['High'] + series['Low'] + series['SMA'])/3
    
    return (num - series['SMA']) / (0.015*series['SD'])


def compute_adx(series,trail_size = 20):
    """A function to take in a dataframe for one stock time series,
    and return a panda series Average Directional Index."""
    
    return series['DX'].rolling(window=trail_size, min_periods=1).mean()


def add_technical_markers(series, trail_size=20):
    series['HH'] = compute_highest_high(series, trail_size)
    series['LL'] = compute_lowest_low(series, trail_size)
    series['AV'] = compute_avg_vol(series, trail_size)
    series['SMA'] = compute_sma(series, trail_size)
    series['SD'] = compute_sd(series, trail_size)
    series['WILLR'] = compute_willr(series)
    series['ATR'] = compute_atr(series, trail_size)
    series['DMH'] = compute_dmh(series)
    series['DML'] = compute_dml(series)
    series['EMA'] = compute_ema(series, trail_size)
    series['WMA'] = compute_wma(series, trail_size)
    series['BBHIGH'] = compute_bbhigh(series, trail_size)
    series['BBLOW'] = compute_bblow(series, trail_size)
    series['PERBHIGH'] = compute_perbhigh(series, trail_size)
    series['PERBLOW'] = compute_perblow(series, trail_size)
    series['TRIMA'] = compute_trima(series, trail_size)
    series['RSI'] = compute_rsi(series, trail_size)
    series['DX'] = compute_dx(series, trail_size)
    series['PDI'] = compute_positive_di(series, trail_size)
    series['NDI'] = compute_negative_di(series, trail_size)
    series['ADX'] = compute_adx(series,trail_size)
    series['ROC'] = compute_roc(series,trail_size)
    series['MACD'] = compute_macd(series,trail_size)
    series['CCI'] = compute_cci(series,trail_size)
    return series












