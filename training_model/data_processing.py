import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Drop "SWE" only for features
    features = data.drop(columns=['SWE'])
    numerical_features = features.drop(columns=['date']) # only consider numerical features for training
    
    # extract target
    target = data['SWE']
    
    # normalizing the data
    # Let's make sure to denormalize the data later when done
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(numerical_features)
    
    # perform train test split function
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    
    # Create tensor objects for train test split data
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    # Return the train test splits, the target and features for training
    return X_train, X_test, y_train, y_test, scaler, features, numerical_features, target
