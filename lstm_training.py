import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

#define the LTSM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out  

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
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    # Adjust hyperparameters
    input_size = X_train.shape[1]
    hidden_size = 50
    num_layers = 2
    output_size = 1
    num_epochs = 10000
    batch_size = 32
    learning_rate = 0.001

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #Initialize the model, loss function, and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch.unsqueeze(1))
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}') 

    # Evaluate the model
    model.eval()

    target_scaler = MinMaxScaler()
    target_scaler.fit_transform(target.values.reshape(-1, 1))
    with torch.no_grad():
        predictions = model(X_test.unsqueeze(1)).numpy()
        predictions_original_scale = target_scaler.inverse_transform(predictions)
    
    # Calculate performance metrics
    mae = mean_absolute_error(y_test, predictions_original_scale)
    mse = mean_squared_error(y_test, predictions_original_scale)
    rmse = np.sqrt(mse)

    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

    #Create a DataFrame to store the predictions
    predictions_df = pd.DataFrame(X_test.numpy(), columns=numerical_features.columns)
    predictions_df['SWE_Actual'] = y_test.numpy()
    predictions_df['SWE_Predicted'] = predictions_original_scale

    #Save the predictions to a CSV file   
    predictions_formatted_df = pd.DataFrame({
        'Date': features['date'],
        'Latitude': numerical_features['lat'],
        'Longitude': numerical_features['lon'],
        #'SWE_Actual': predictions_df['SWE_Actual'],
        'SWE_Predicted': predictions_df['SWE_Predicted']
    })

    predictions_formatted_df.to_csv("swe_predictions.csv", index=False)

    print("SWE Predictions along with actual values saved to swe_predictions.csv")

if __name__ == '__main__':
    main()
