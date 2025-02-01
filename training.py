import pandas as pd
import lightgbm as lgb
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Define meteorological variables
variables = ['precip', 'Rmax', 'Rmin', 'SPH', 'SRAD', 'tmax', 'tmin', 'windspeed']

# Load meteorological data
meteo_dfs = []
# for var in variables:
#     file_path = f"input_data/meteorological_data/Modified_Output_{var}.csv"
#     df = pd.read_csv(file_path).rename(columns={
#         'lat': 'latitude', 
#         'lon': 'longitude',
#         'variable_value': 'value'})
#     df['variable'] = var
#     meteo_dfs.append(df)

# test 1 variable
file_path = f"input_data/meteorological_data/Modified_Output_windspeed.csv"
df = pd.read_csv(file_path).rename(columns={
    'lat': 'latitude', 
    'lon': 'longitude',
    'variable_value': 'value'})
df['variable'] = 'windspeed'
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

# Define features and target
# features = ['precip', 'Rmax', 'Rmin', 'SPH', 'SRAD', 'tmin', 'windspeed', 'Elevation', 'Southness']
features = ['Elevation', 'Southness', 'windspeed', 'latitude', 'longitude', 'date']
target = 'SWE'

# Preprocessing function
def preprocess(df):
    required_cols = features + [target]
    df = df[required_cols].copy()
    for col in features:
        if col != 'date' and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df[features]

# Time-series split
# merged_data = merged_data.sort_values('date')
# train_size = int(len(merged_data) * 0.8)
# X_train = merged_data.iloc[:train_size][features]
# y_train = merged_data.iloc[:train_size][target]

# Prepare data for modeling
X = preprocess(merged_data)
y = merged_data[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_features = ['windspeed']
model = lgb.LGBMRegressor()
model.fit(X_train[model_features], y_train)

# Evaluate model
train_predictions = model.predict(X_train[model_features])
test_predictions = model.predict(X_test[model_features])

train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': model_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# test
test_predictions_df = pd.DataFrame({
    "Date": X_test["date"],
    "Latitude": X_test["latitude"],
    "Longitude": X_test["longitude"],
    "SWE_prediction": test_predictions
})

print("\nSample predictions:")
print(test_predictions_df)
test_predictions_df.to_csv("swe_predictions.csv", index=False)

# # Predict test locations
# test_dynamic = pd.read_csv('additional_test_locations/Test_InputData_dynamicVars_2017_2019.csv')
# test_static = pd.read_csv('additional_test_locations/Test_InputData_staticVars_2017_2019.csv')
# test_data = test_dynamic.merge(test_static, on=['latitude', 'longitude'])
# # test_processed = preprocess(test_data)

# test_processed = test_data[features]
# predictions = model.predict(test_processed)

# # Save predictions
# output = pd.DataFrame({
#     "Date": test_data["date"],
#     "Latitude": test_data["latitude"],
#     "Longitude": test_data["longitude"],
#     "SWE_prediction": predictions
# }).to_csv("swe_predictions.csv", index=False)
# print(output)
# output.to_csv("swe_predictions.csv", index=False)
