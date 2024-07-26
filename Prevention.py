import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Sample training data
data = {
    'timestamp': [
        '2024-07-21 10:00:00', '2024-07-21 10:01:00', '2024-07-21 10:02:00', 
        '2024-07-21 10:03:00', '2024-07-21 10:04:00'
    ],
    'latitude': [37.7749, 37.7750, 37.7751, 37.7752, 37.7753],
    'longitude': [-122.4194, -122.4195, -122.4196, -122.4197, -122.4198],
    'is_spoofed': [0, 0, 1, 0, 1]  # Example labels (0 for real, 1 for spoofed)
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
df['timestamp_diff'] = df['timestamp'].diff().dt.total_seconds()

# Fill any NaN values resulting from the shift operation
df = df.fillna(0)

# Prepare features and labels
features = df[['distance', 'speed', 'timestamp_diff']]
labels = df['is_spoofed']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

def detect_spoofing(gps_data):
    gps_data['timestamp'] = pd.to_datetime(gps_data['timestamp'])
    gps_data['distance'] = haversine(gps_data['longitude'].shift(), gps_data['latitude'].shift(), gps_data['longitude'], gps_data['latitude'])
    gps_data['speed'] = gps_data['distance'] / (gps_data['timestamp'].diff().dt.total_seconds() / 3600)
    gps_data['timestamp_diff'] = gps_data['timestamp'].diff().dt.total_seconds()
    gps_data = gps_data.fillna(0)
    features = gps_data[['distance', 'speed', 'timestamp_diff']]
    prediction = model.predict(features)
    gps_data['is_spoofed'] = prediction
    return gps_data

# Example usage
new_data = pd.read_csv('new_gps_data.csv')
result = detect_spoofing(new_data)
print(result)