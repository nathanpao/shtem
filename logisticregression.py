import os
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

results = []

dir = r"C:\Users\natha\Downloads\S&Ptestabbrv"
os.chdir(dir)

for filename in os.listdir(dir):
    if ".csv" in filename:
        Stock_Name = ((os.path.basename(filename)).split(".csv")[0])
        data = pd.read_csv(os.path.join(filename))
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(0)
        dependents = [data["five_minute_Observation_Outcome"].to_list(), data["thirty_minute_Observation_Outcome"].to_list(),
                      data["sixty_minute_Observation_Outcome"].to_list()]
        data = data.drop(
            ['five_minute_Observation_Outcome', 'thirty_minute_Observation_Outcome', 'sixty_minute_Observation_Outcome', 'Date',
             'Open', 'High', 'Low', 'Close'], axis=1)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        results_Section = []
        p = 0
        for dep in dependents:
            x_train, x_test, y_train, y_test = train_test_split(data, dep, test_size=0.2, random_state=0)
            model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr', random_state=0)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)  # To get the predicted values
            conf = confusion_matrix(y_test, y_pred)
            if p == 0:
                results.append(
                    [Stock_Name, "five_minute_Observation_Outcome", model.score(x_train, y_train), model.score(x_test, y_test),
                     conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]])
            if p == 1:
                results.append([Stock_Name, "thirty_minute_Observation_Outcome", model.score(x_train, y_train),
                                     model.score(x_test, y_test), conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]])
            if p == 2:
                results.append([Stock_Name, "sixty_minute_Observation_Outcome", model.score(x_train, y_train),
                                     model.score(x_test, y_test), conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]])
            p += 1
        print("Model complete for " + Stock_Name)
df = pd.DataFrame(results, columns=['Stock', 'Observation Period', 'Model Accuracy on Training Data',
                                         'Model Accuracy on Test Data', 'True Positives', 'False Positives',
                                         'False Negative', 'True Negative'])
df.to_csv("model.csv")

''' for f in os.listdir(filename):
        if "aapl" in f:
            Stock_Name = ((os.path.basename(f)).split(".csv")[0])
            data = pd.read_csv(os.path.join(filename, f), header=None)
            data.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'X', 'Y', 'Z']
            dropna(data)
            data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
            data = data.iloc[100:]
            close_prices = data['Close'].tolist()
            five_minute_Obs = []
            thirty_minute_Obs = []
            sixty_minute_Obs = []
            x = 0
            while x < (len(data)):
                if x < (len(data) - 5):
                    if ((close_prices[x + 1] + close_prices[x + 2] + close_prices[x + 3] + close_prices[x + 4] +
                         close_prices[x + 5]) / 5) > close_prices[x]:
                        five_minute_Obs.append(1)
                    else:
                        five_minute_Obs.append(0)
                else:
                    five_minute_Obs.append(0)
                x += 1
            y = 0
            while y < (len(data)):
                if y < (len(data) - 30):
                    thirtyMinuteCalc = 0
                    y2 = 0
                    while y2 < 30:
                        thirtyMinuteCalc = thirtyMinuteCalc + close_prices[y + y2]
                        y2 += 1
                    if (thirtyMinuteCalc / 30) > close_prices[y]:
                        thirty_minute_Obs.append(1)
                    else:
                        thirty_minute_Obs.append(0)
                else:
                    thirty_minute_Obs.append(0)
                y += 1
            z = 0
            while z < (len(data)):
                if z < (len(data) - 60):
                    sixtyMinuteCalc = 0
                    z2 = 0
                    while z2 < 60:
                        sixtyMinuteCalc = sixtyMinuteCalc + close_prices[z + z2]
                        z2 += 1
                    if (sixtyMinuteCalc / 60) > close_prices[z]:
                        sixty_minute_Obs.append(1)
                    else:
                        sixty_minute_Obs.append(0)
                else:
                    sixty_minute_Obs.append(0)
                z += 1
            data['five_minute_Observation_Outcome'] = five_minute_Obs
            data['thirty_minute_Observation_Outcome'] = thirty_minute_Obs
            data['sixty_minute_Observation_Outcome'] = sixty_minute_Obs
            data.to_csv(Stock_Name + str(data.iloc[0, 0]) + "model.csv")
            print("Data for " + Stock_Name + " has been substantiated with technical features.")'''