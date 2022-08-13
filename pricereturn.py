import os
import pandas as pd

dir = r"C:\Users\natha\Downloads\S&Ptest"
os.chdir(dir)

dates = []
returns = []

for filename in os.listdir(dir):
    dayReturns = {}
    for f in os.listdir(filename):
        stockName = str(f)[6:-4].upper()
        try:
            stock = pd.read_csv(os.path.join(filename, f), header=None)
        except:
            continue
        try:
            date = int(stock.iloc[0, 0])
            open = float(stock.iloc[0, 2])
            close = float(stock.iloc[-1, 5])
            returnValue = (close - open)/open
            dayReturns[stockName] = returnValue
        except:
            date = 0
            dayReturns[stockName] = 0
    dates.append(date)
    returns.append(dayReturns)

df = pd.DataFrame.from_dict(returns, orient='columns')
df.index = [dates]
df.to_csv(r"C:\Users\natha\Downloads\stockreturns.csv")
