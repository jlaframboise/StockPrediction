import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def apply_rolling(stock, trail_size, predict_length):
    x = []
    y = []
    for i in range(trail_size, len(stock) + 1 - predict_length):
        x_point = stock.drop(columns=['Date', 'Ticker']).iloc[i-trail_size : i].values
        y_point = stock['Close'].iloc[i + predict_length -1]
        
        if np.isnan(x_point).sum() ==0:
            x.append(x_point)
            y.append(y_point)
    
    return np.array(x), np.array(y)


def roll_all_stocks(dataset, trail_size, predict_length):
    res = dataset.groupby('Ticker').apply(lambda x: apply_rolling(x, trail_size=trail_size, predict_length=predict_length))
    x = [x[0] for x in res.values]
    y = [x[1] for x in res.values]
    return np.concatenate(x), np.concatenate(y)


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
