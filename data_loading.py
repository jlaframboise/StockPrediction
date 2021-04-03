import pandas as pd
import numpy as np
import yfinance as yf

def process_stock_data_from_yf(data):
    """
    A function to take in the data from yahoo finance
    retrieved by yfinance, 
    and it will convert it from a wide format to a long format
    which will allow easier manipulation for our workflows. 
    """
    
    df = pd.melt(data, ignore_index=False)
    df.columns = ['Variable', 'Ticker', 'Value']
    # this is a nicely formed dataframe of historical data
    nice_df = df.reset_index().pivot(index=['Ticker', 'Date'], columns=['Variable'], values=['Value'])
    nice_df.columns = nice_df.columns.droplevel()
    df = nice_df.reset_index()
    df.columns.name=None
    return df

def get_data_yf(tickers, time_period):
    """
    A wrapper function for yf.download 
    that abstracts parameters and 
    joins the list of tickers. 
    """
    data = yf.download(tickers = ' '.join(tickers),
                   auto_adjust=True,
                  period=time_period)
    return data

def filter_has_all_data(df):
    """
    A function that will take in a dataset and for each stock:
    check if that stock has data for the whole time period studied. 

    We do not want to have stocks that only have data for the 
    last 2 years because then they will only be in our testing set. 

    Returns the cleaned df, and the percentage of stocks that 
    did not get removed. 
    """
    tickers_with_all_data = df.groupby("Ticker").apply(lambda x: x.isna().sum()).max(axis=1)==0
    percent = tickers_with_all_data.mean()*100
    tickers_with_all_data_list = tickers_with_all_data[tickers_with_all_data].index.tolist()
    return (percent, df[df['Ticker'].isin(tickers_with_all_data_list)].reset_index(drop=True) )