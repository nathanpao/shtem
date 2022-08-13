import pandas as pd
import matplotlib.pyplot as plt

stock = pd.read_csv(r"C:\Users\natha\Downloads\S&Ptest\allstocks_20190319\table_aapl.csv", header=None)

#plots minimums and maximums
stock.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'X', 'Y', 'Z']
stock['min'] = stock.Close[(stock.Close.shift(1) > stock.Close) & (stock.Close.shift(-1) > stock.Close)]
stock['max'] = stock.Close[(stock.Close.shift(1) < stock.Close) & (stock.Close.shift(-1) < stock.Close)]

plt.scatter(stock.index, stock['min'], color = 'red')
plt.scatter(stock.index, stock['max'], color = 'green')
stock.Close.plot()
plt.savefig(r"C:\Users\natha\Downloads\minmaxplot.png")

#records minimums and maximums as a dictionary
minimums = {}
maximums = {}

for i in range(1, stock.shape[0]-1):
    if (stock.iloc[i+1, 5] > stock.iloc[i, 5]) & (stock.iloc[i-1, 5] > stock.iloc[i, 5]):
        minimums[stock.iloc[i, 1]] = stock.iloc[i, 5]
    if (stock.iloc[i + 1, 5] < stock.iloc[i, 5]) & (stock.iloc[i - 1, 5] < stock.iloc[i, 5]):
        maximums[stock.iloc[i, 1]] = stock.iloc[i, 5]

print(minimums)
print(maximums)
