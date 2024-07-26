import pandas as pd
import numpy as np

# Sample data
data = {
    'timestamp': [
        '2024-07-21 10:00:00', '2024-07-21 10:01:00', '2024-07-21 10:02:00', 
        '2024-07-21 10:03:00', '2024-07-21 10:04:00'
    ],
    'latitude': [37.7749, 37.7750, 37.7751, 37.7752, 37.7753],
    'longitude': [-122.4194, -122.4195, -122.4196, -122.4197, -122.4198]
}
df = pd.DataFrame(data)

# Convert the 'timestamp' column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

def haversine(lon1, lat1, lon2, lat2):
    # Calculate the great-circle distance between two points
    R = 6371  # Radius of the Earth in km
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Calculate distances between consecutive points
df['distance'] = haversine(df['longitude'].shift(), df['latitude'].shift(), df['longitude'], df['latitude'])

# Calculate speed in km/h
df['speed'] = df['distance'] / (df['timestamp'].diff().dt.total_seconds() / 3600)

# Calculate time differences
df['timestamp_diff'] = df['timestamp'].diff()

print(df.head())
