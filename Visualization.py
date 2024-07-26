import pandas as pd
import folium

# Load data
data = pd.read_csv('C:\Users\jayakumar\Downloads\csv file geolite')

# Print columns to understand the data
print("Columns in the dataset:")
print(data.columns)

# Inspect the first few rows to understand the data structure
print("Sample data:")
print(data.head())

# Identify the correct column names for latitude and longitude
# You might need to adjust these names based on the actual columns in your dataset
latitude_col = 'latitude'  # Change if the actual column name is different
longitude_col = 'longitude'  # Change if the actual column name is different
is_spoofed_col = 'is_spoofed'  # This column might not exist, you may need to add it or adjust accordingly

# Check if the latitude and longitude columns exist and handle accordingly
if latitude_col in data.columns and longitude_col in data.columns:
    m = folium.Map(location=[data[latitude_col].mean(), data[longitude_col].mean()], zoom_start=12)

    # If 'is_spoofed' column doesn't exist, add a placeholder for testing
    if is_spoofed_col not in data.columns:
        data[is_spoofed_col] = 0  # Assuming non-spoofed by default if column doesn't exist

    # Add points to the map
    for _, row in data.iterrows():
        color = 'red' if row[is_spoofed_col] else 'blue'
        folium.CircleMarker(
            location=[row[latitude_col], row[longitude_col]],
            color=color,
            radius=3,
            fill=True,
            fill_color=color
        ).add_to(m)

    # Save map
    m.save('gps_map.html')

    print("Map has been created and saved as 'gps_map.html'.")
else:
    print(f"Required columns '{latitude_col}' or '{longitude_col}' are missing in the dataset. Please check the column names and update accordingly.")