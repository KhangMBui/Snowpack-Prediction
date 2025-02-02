import pandas as pd
from sklearn.neighbors import BallTree


# Define meteorological variables
variables = ['precip', 'Rmax', 'Rmin', 'SPH', 'SRAD', 'tmax', 'tmin', 'windspeed']

# Load meteorological data
meteo_dfs = []
for var in variables:
    file_path = f"input_data/meteorological_data/Modified_Output_{var}.csv"
    df = pd.read_csv(file_path).rename(columns={
        'lat': 'latitude', 
        'lon': 'longitude',
        'variable_value': 'value'})
    df['variable'] = var
    meteo_dfs.append(df)

merged_meteo = pd.concat(meteo_dfs)

# Pivot to wide format (one column per variable)
merged_meteo = merged_meteo.pivot_table(
    index=['date', 'latitude', 'longitude'],
    columns='variable',
    values='value',
    aggfunc='first' #duplication guard
).reset_index()

# Load station data
station_info = pd.read_csv("input_data/swe_data/Station_Info.csv").rename(columns={
    'Latitude': 'latitude',
    'Longitude': 'longitude'
})
swe_value = pd.read_csv("input_data/swe_data/SWE_values_all.csv").rename(columns={
    'Date': 'date',
    'Latitude': 'latitude',
    'Longitude': 'longitude'
})

# Spatial join to link stations with grid points
grid_points = merged_meteo[['latitude', 'longitude']].drop_duplicates().values

tree = BallTree(grid_points, leaf_size=2)
stations = station_info[['latitude', 'longitude']].values
_, indices = tree.query(stations, k=1)
station_to_grid = grid_points[indices.flatten()]

# Create grid mapping DataFrame
grid_mapping = pd.DataFrame({
    'station_lat': station_info['latitude'],
    'station_lon': station_info['longitude'],
    'grid_lat': station_to_grid[:, 0],
    'grid_lon': station_to_grid[:, 1],
    'Elevation': station_info['Elevation'], 
    'Southness': station_info['Southness']
})

# Merge all data
merged_data = (
    swe_value
    .merge(grid_mapping, left_on=['latitude', 'longitude'], right_on=['station_lat', 'station_lon'])
    .merge(
        merged_meteo,
        left_on=['date', 'grid_lat', 'grid_lon'],
        right_on=['date', 'latitude', 'longitude'],
        suffixes=('_station', '_grid')  # Add suffixes to resolve conflicts
    )
    # Rename columns to retain latitude/longitude
    .rename(columns={
        'latitude_grid': 'latitude',
        'longitude_grid': 'longitude'
    })
    .drop(columns=['latitude_station', 'longitude_station'])
)

# Drop unnecessary columns
merged_data = merged_data.drop(columns=['station_lat', 'station_lon', 'grid_lat', 'grid_lon'])

# Ensure 'date' is in datetime format for proper sorting
merged_data['date'] = pd.to_datetime(merged_data['date'])

# Define the preferred column order
column_order = ['date','latitude', 'longitude', 'SWE', 'precip', 'tmin', 'tmax',
                 'SPH', 'SRAD', 'Rmax', 'Rmin', 'windspeed', 'Elevation', 'Southness' ]

# Sort by date in ascending order
merged_data = merged_data.sort_values(by='date', ascending=True)

# Output to CSV file
merged_data[column_order].to_csv("./merged_data.csv", index=False)
