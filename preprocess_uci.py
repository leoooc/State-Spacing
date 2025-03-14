
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("AirQualityUCI/AirQualityUCI.csv", sep=";", decimal=",", parse_dates=[['Date', 'Time']], infer_datetime_format=True)

df = df.iloc[:, :-2]
df.replace(-200, np.nan, inplace=True)
df.fillna(method='ffill', inplace=True)
df.set_index('Date_Time', inplace=True)

data = df.values

scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

preprocessed_df = pd.DataFrame(data_normalized, index=df.index, columns=df.columns)

output_file = "preprocessed_airquality.csv"
preprocessed_df.to_csv(output_file)

print(f"Preprocessing complete. Data saved to {output_file}")
