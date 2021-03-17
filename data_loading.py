import pandas as pd
import numpy as np
import yfinance as yf

def process_stock_data_from_yf(data):
    
    df = pd.melt(data, ignore_index=False)
    df.columns = ['Variable', 'Ticker', 'Value']
    # this is a nicely formed dataframe of historical data
    nice_df = df.reset_index().pivot(index=['Ticker', 'Date'], columns=['Variable'], values=['Value'])
    nice_df.columns = nice_df.columns.droplevel()
    df = nice_df.reset_index()
    df.columns.name=None
    return df

def get_data_yf(tickers, time_period):
    data = yf.download(tickers = ' '.join(tickers),
                   auto_adjust=True,
                  period=time_period)
    return data

def filter_has_all_data(df):
    tickers_with_all_data = df.groupby("Ticker").apply(lambda x: x.isna().sum()).max(axis=1)==0
    percent = tickers_with_all_data.mean()*100
    tickers_with_all_data_list = tickers_with_all_data[tickers_with_all_data].index.tolist()
    return (percent, df[df['Ticker'].isin(tickers_with_all_data_list)].reset_index(drop=True) )