
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
# The UCI Air Quality dataset is semicolon-separated and uses comma as the decimal.
df = pd.read_csv("AirQualityUCI/AirQualityUCI.csv", sep=";", decimal=",", parse_dates=[['Date', 'Time']], infer_datetime_format=True)

# The dataset often has two trailing empty columns; drop them.
df = df.iloc[:, :-2]

# Replace known error values (-200) with NaN
df.replace(-200, np.nan, inplace=True)

# Fill missing values using forward-fill (you can adjust this if needed)
df.fillna(method='ffill', inplace=True)

# Set the combined date_time as index (adjust column name if necessary)
df.set_index('Date_Time', inplace=True)

# (Optional) Drop columns that are not sensor readings.
# For example, if there's any column that you don't need for modeling, drop it here.
# For now, we keep all available sensor data.

# Convert DataFrame to numpy array
data = df.values

# Normalize the data (important for many state-space models)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Convert back to DataFrame with original index and columns
preprocessed_df = pd.DataFrame(data_normalized, index=df.index, columns=df.columns)

# Save the preprocessed data to a CSV file that train.py can load
output_file = "preprocessed_airquality.csv"
preprocessed_df.to_csv(output_file)

print(f"Preprocessing complete. Data saved to {output_file}")