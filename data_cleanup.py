import os
import pandas as pd

# Define file paths
input_dir = "./input_data"
meteo_dir = os.path.join(input_dir, "meteorological_data")
swe_file = os.path.join(input_dir, "swe_data", "SWE_values_all.csv")
station_file = os.path.join(input_dir, "swe_data", "Station_Info.csv")