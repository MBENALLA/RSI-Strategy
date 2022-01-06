import pandas as pd
import numpy as np
import talib
from binance.client import Client
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
SPOT_TRADE_FEES = 0.00075

client = Client()


def get_data(coin, back):
    df = pd.DataFrame(client.get_historical_klines(coin,
                                                   Client.KLINE_INTERVAL_5MINUTE,
                                                   str(back) + ' days ago UTC',
                                                   '5 min ago UTC'))

    closes = pd.DataFrame(df[4])
    closes.columns = ['Close']
    closes.Close = closes.Close.astype(float)
    return closes


engine = create_engine('sqlite:///backtestRSI_db.db')
# engine.execute("DELETE FROM COIN_TABLE")
# get_data('FTMBUSD', 30).to_sql('COIN_TABLE', engine, if_exists='append', index=False)
data = pd.read_sql('COIN_TABLE', engine)


def rsi_strat(closes, in_position=False):
    closes = closes.copy()
    positions = []
    np_closes = np.asarray(closes.Close)
    closes['RSI'] = talib.RSI(np_closes)
    closes.dropna(inplace=True)
    for index, row in closes.iterrows():
        if row['RSI'] < RSI_OVERSOLD and not in_position:
            positions.append(1)
            in_position=True
        elif row['RSI'] > RSI_OVERBOUGHT and in_position:
            positions.append(1)
            in_position=False
        else:
            positions.append(0)
    closes['Position'] = positions
    closes['v_Position'] = closes['Position'] * closes['Close']
    closes = closes[closes['Position'] != 0]
    return closes


def back_test(dataframe):
    dataframe['ROC'] = dataframe.v_Position.pct_change()
    np_roc = np.asarray(dataframe['ROC'])
    roc_size = len(np_roc)
    perf = 0
    if (roc_size % 2) == 0:
        for i in range(0, (roc_size - 1), 2):
            perf = perf+np_roc[i+1]
    else:
        for i in range(0, (roc_size - 2), 2):
            perf = perf+np_roc[i+1]
    trade_fees = roc_size * SPOT_TRADE_FEES
    return roc_size, trade_fees, perf


df1 = rsi_strat(data)
trades, fees, performance = back_test(df1)
net_profit = round((performance-fees)*100, 4)
print('Trades: {} | Fees: {}'.format(trades, fees))
print('Performance: {}% | Net profit: {}%'.format(round(performance*100, 4), net_profit))
