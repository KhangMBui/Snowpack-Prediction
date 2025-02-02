import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def main():
    
    # Read data frame from combined and cleaned data
    data = pd.read_csv("./input_data/cleaned_data/combined_dataset.csv")
    
    features = data.drop(columns=['SWE'])
    numerical_features = features.drop(columns=['date'])
    target = data['SWE']
    
    #normalize the data
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(numerical_features)

    #Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    #Convert the data into tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

if __name__ == '__main__':
    main()