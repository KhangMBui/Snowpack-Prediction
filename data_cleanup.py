import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist

# 1. Load the datasets
def load_data():
    # Load meteorological data
    windspeed = pd.read_csv('./input_data/meteorological_data/Modified_Output_windspeed.csv')
    precip = pd.read_csv('./input_data/meteorological_data/Modified_Output_precip.csv')
    srad = pd.read_csv('./input_data/meteorological_data/Modified_Output_SRAD.csv')
    tmin = pd.read_csv('./input_data/meteorological_data/Modified_Output_tmin.csv')
    tmax = pd.read_csv('./input_data/meteorological_data/Modified_Output_tmax.csv')
    rmin = pd.read_csv('./input_data/meteorological_data/Modified_Output_Rmin.csv')
    rmax = pd.read_csv('./input_data/meteorological_data/Modified_Output_Rmax.csv')
    sph = pd.read_csv('./input_data/meteorological_data/Modified_Output_SPH.csv')

    # Load SWE data and Station Info
    station_info = pd.read_csv('./input_data/swe_data/Station_Info.csv')
    swe_values = pd.read_csv('./input_data/swe_data/SWE_values_all.csv')

    return windspeed, precip, srad, tmin, tmax, rmin, rmax, sph, station_info, swe_values

# 2. Handle Missing Values (Imputation)
def handle_missing_values(windspeed, precip, srad, tmin, tmax, rmin, rmax, sph):
    # Use mean imputation for missing values
    imputer = SimpleImputer(strategy='mean')
    
    # Impute missing values in each meteorological dataset
    windspeed['variable_value'] = imputer.fit_transform(windspeed[['variable_value']])
    precip['variable_value'] = imputer.fit_transform(precip[['variable_value']])
    srad['variable_value'] = imputer.fit_transform(srad[['variable_value']])
    tmin['variable_value'] = imputer.fit_transform(tmin[['variable_value']])
    tmax['variable_value'] = imputer.fit_transform(tmax[['variable_value']])
    rmin['variable_value'] = imputer.fit_transform(rmin[['variable_value']])
    rmax['variable_value'] = imputer.fit_transform(rmax[['variable_value']])
    sph['variable_value'] = imputer.fit_transform(sph[['variable_value']])
    
    return windspeed, precip, srad, tmin, tmax, rmin, rmax, sph
# 3. Spatial Association of SNOTEL Locations to Grids
def associate_snotel_to_grids(station_info, meteorological_datasets):
    # Extract latitudes and longitudes for both SNOTEL stations and grid points
    snotel_locations = station_info[['Latitude', 'Longitude']].values
    
    # Initialize a dictionary to store closest grid indices for each meteorological dataset
    closest_grid_indices = {}

    for dataset_name, dataset in meteorological_datasets.items():
        grid_points = dataset[['lat', 'lon']].values  # Assuming all datasets have the same grid points

        # Calculate Euclidean distance between SNOTEL stations and grid points
        distances = cdist(snotel_locations, grid_points, metric='euclidean')
        
        # Find the closest grid point for each SNOTEL station
        closest_grid_indices[dataset_name] = np.argmin(distances, axis=1)
        
        # Debugging: Print the shape and first few distances for validation
        # print(f"Distances for {dataset_name}: {distances.shape}")
        # print(f"Closest grid indices for {dataset_name}: {closest_grid_indices[dataset_name][:5]}")
    
    # Add the closest grid indices for each dataset to the station_info DataFrame
    for dataset_name, indices in closest_grid_indices.items():
        station_info[f'closest_grid_index_{dataset_name}'] = indices
    
    return station_info

# 4. Combine Data (Attach Static and Meteorological Features to Each SNOTEL Station)
def combine_data(station_info, meteorological_datasets, swe_values):
    # Merge station_info with swe_values based on Latitude and Longitude
    station_info = pd.merge(station_info, swe_values[['Latitude', 'Longitude', 'SWE']], on=['Latitude', 'Longitude'], how='left')
    
    # Create a combined dataframe for each SNOTEL station
    combined_data = []

    for index, station in station_info.iterrows():
        # Get the closest grid indices for each dataset
        combined_row = {
            'date': None,  # We will fill this in based on the data
            'lat': station['Latitude'],
            'lon': station['Longitude'],
            'SWE': station['SWE'],
            'elevation': station['Elevation'],
            'southness': station['Southness']
        }
        
        for dataset_name, dataset in meteorological_datasets.items():
            grid_index = station[f'closest_grid_index_{dataset_name}']
            grid_data = dataset.iloc[grid_index]
            
            # Extract data for the nearest grid point
            combined_row[f'{dataset_name}_value'] = grid_data['variable_value']
            if combined_row['date'] is None:
                combined_row['date'] = grid_data['date']  # Assume all datasets share the same date
        
        # Append the combined row to the list
        combined_data.append(combined_row)

    # Convert the list of combined data into a DataFrame
    combined_df = pd.DataFrame(combined_data)
    
    # Drop any duplicate rows and reset the index
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    # Make sure the 'date' column is properly sorted to preserve the full time range
    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
    
    # Replace invalid dates (NaT) with "NAT"
    combined_df['date'] = combined_df['date'].fillna("NAT")
    
    combined_df = combined_df.sort_values(by='date', ascending=True)

    return combined_df

# Main function to execute the preprocessing
def main():
    # Load data
    windspeed, precip, srad, tmin, tmax, rmin, rmax, sph, station_info, swe_values = load_data()
    
    # Handle missing values
    meteorological_datasets = {
        'windspeed': windspeed,
        'precip': precip,
        'srad': srad,
        'tmin': tmin,
        'tmax': tmax,
        'rmin': rmin,
        'rmax': rmax,
        'sph': sph
    }
    
    windspeed, precip, srad, tmin, tmax, rmin, rmax, sph = handle_missing_values(windspeed, precip, srad, tmin, tmax, rmin, rmax, sph)
    
    # Spatial association of SNOTEL stations to grids
    station_info = associate_snotel_to_grids(station_info, meteorological_datasets)
    
    # Combine data for each SNOTEL location
    combined_data = combine_data(station_info, meteorological_datasets, swe_values)
    
    # Save the combined dataset to a CSV
    combined_data.to_csv('combined_dataset.csv', index=False)
    
    print("Preprocessing complete. Combined dataset saved to 'combined_dataset.csv'.")

if __name__ == "__main__":
    main()
