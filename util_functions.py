import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def apply_rolling(stock, trail_size, predict_length, predict_change=False, trend_classify=False):
    x = []
    y = []
    tickers = []
    for i in range(trail_size, len(stock) + 1 - predict_length):
        x_point = stock.drop(columns=['Date', 'Ticker']).iloc[i-trail_size : i].values
        if trend_classify:
            y_point = 1 if stock['Close'].iloc[i + predict_length -1] > stock['Close'].iloc[i -1] else 0
        elif predict_change:
            y_point = stock['Close'].iloc[i + predict_length -1] - stock['Close'].iloc[i -1]
        else:
            y_point = stock['Close'].iloc[i + predict_length -1]
        ticker = stock['Ticker'].iloc[i + predict_length -1]
        
        if np.isnan(x_point).sum() ==0:
            x.append(x_point)
            y.append(y_point)
            tickers.append(ticker)
    
    return np.array(x), np.array(y), np.array(tickers)

def split_and_apply_rolling(stock, trail_size, predict_length, hist_features, tech_features):
    xh = []
    xt = []
    y = []
    tickers = []
    for i in range(trail_size, len(stock) + 1 - predict_length):
        # historical data from t-trail_size to t-1 inclusive
        xh_point = stock.drop(columns=['Date', 'Ticker']+tech_features).iloc[i-trail_size : i].values
        # technical data at time t-1
        xt_point = stock[tech_features].iloc[i - 1]
        # label at time t-1 + predict_length
        y_point = stock['Close'].iloc[i + predict_length -1]
        ticker = stock['Ticker'].iloc[i + predict_length -1]
        
        if np.isnan(xh_point).sum() ==0 and np.isnan(xt_point).sum() ==0:
            xh.append(xh_point)
            xt.append(xt_point)
            y.append(y_point)
            tickers.append(ticker)
    
    return np.array(xh), np.array(xt), np.array(y), np.array(tickers)

def split_and_roll_all_stocks(dataset, trail_size, predict_length, hist_features, tech_features):
    res = dataset.groupby('Ticker').apply(lambda x: split_and_apply_rolling(x, trail_size=trail_size, predict_length=predict_length, hist_features=hist_features, tech_features=tech_features))
    xh = [x[0] for x in res.values]
    xt = [x[1] for x in res.values]
    y = [x[2] for x in res.values]
    tickers = [x[3] for x in res.values]
    return np.concatenate(xh), np.concatenate(xt), np.concatenate(y), np.concatenate(tickers)

def roll_all_stocks(dataset, trail_size, predict_length, predict_change=False, trend_classify=False):
    res = dataset.groupby('Ticker').apply(lambda x: apply_rolling(x, trail_size=trail_size, predict_length=predict_length, predict_change=predict_change, trend_classify=trend_classify))
    x = [x[0] for x in res.values]
    y = [x[1] for x in res.values]
    tickers = [x[2] for x in res.values]
    x = np.concatenate(x)
    y = np.concatenate(y)
    tickers = np.concatenate(tickers)
    return x, y, tickers





def evaluate_model_rmse(y_preds, y_true, num_features, scaler):
    dummies = np.zeros((y_preds.shape[0], num_features-1))
    res = np.concatenate([y_preds, dummies], axis=1)
    pred_dollars = scaler.inverse_transform(res)[:, 0]
    
    res2 = np.concatenate([np.expand_dims(y_true, axis=1), dummies], axis=1)
    true_dollars = scaler.inverse_transform(res2)[:, 0]
    return np.sqrt(mean_squared_error(true_dollars, pred_dollars))


def plot_loss(history):
    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], 'bo--')
    plt.plot(history.history['val_loss'], 'ro-')
    plt.ylabel('Loss')
    plt.xlabel('Epochs (n)')
    plt.legend(['Training loss', 'Validation loss'])
    plt.title("Loss curve for LSTM")
    plt.show()

def plot_acc(history):
    plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'], 'bo--')
    plt.plot(history.history['val_accuracy'], 'ro-')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs (n)')
    plt.legend(['Training accuracy', 'Validation accuracy'])
    plt.title("Accuracy curve for LSTM")
    plt.show()


def norm_per_stock_split(train, valid, test, features, scaler_model):
    """
    Takes in dataframes for train, validation, test 
    with a 'Ticker' column, and will split
    the dataframes by this ticker, then fit_transform the data in train
    with a normalizing model specified in place,
    and transform the data in train and test in place
    and return the mapping of tickers to scaler models. 
    """
    
    scalers = {}

    for ticker in train['Ticker'].unique():
        scaler = scaler_model()
        
        # fit and transform training data
        train.loc[train['Ticker']==ticker, features] = scaler.fit_transform(train.loc[train['Ticker']==ticker, features])
        
        # only transform training and test, do not fit. 
        valid.loc[valid['Ticker']==ticker, features] = scaler.transform(valid.loc[valid['Ticker']==ticker, features])
        test.loc[test['Ticker']==ticker, features] = scaler.transform(test.loc[test['Ticker']==ticker, features])
        
        # save the model that was used for this stock
        scalers[ticker] = scaler
        
    return scalers


def load_climate_data(filenames, terms):
    """
    A function to load in the climate google trends data
    from a list of filenames and a list of corresponding
    terms. 
    """
    # read files
    dfs = [pd.read_csv(f) for f in filenames]
    # rename columns, insert the search term string
    for i in range(len(dfs)):
        dfs[i].columns = ["Date", "Popularity"]
        dfs[i].insert(2, "Term", terms[i])
    # combine to one df
    climate_trends_data = pd.concat(dfs).reset_index(drop=True)
    # convert date col to datetime
    climate_trends_data['Date'] = pd.to_datetime(climate_trends_data['Date'])
    # convert the data from long to wide format
    climate_trends_data = climate_trends_data.pivot(index='Date', columns="Term", values="Popularity").reset_index()
    
    return climate_trends_data

def performance_stats(model, x, y):
    print("Upward ratio: {}".format(np.mean(y)))
    preds = model.predict(x)
    print("Mean prediction: {}".format(np.mean(preds)))
    print("Predicted upward ratio: {}".format(np.mean(preds>0.5)))
    print("Accuracy: {}".format(np.mean( y == [1 if x>0.5 else 0 for x in preds])))