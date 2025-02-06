import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from performance_metrics import nash_sutcliffe_efficiency, relative_bias, actual_error, root_mean_square_error
import pandas as pd

def train_model(model, X_train, y_train, X_test, y_test, scaler, features, numerical_features, target, best_params, num_epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    
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
    
    scaler.fit_transform(target.values.reshape(-1, 1))
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.unsqueeze(1)).numpy()
        predictions_original_scale = scaler.inverse_transform(predictions)
    
    predictions_df = pd.DataFrame(X_test.numpy(), columns=numerical_features.columns)
    predictions_df['SWE_Actual'] = y_test.numpy()
    predictions_df['SWE_Predicted'] = predictions_original_scale
    
    predictions_formatted_df = pd.DataFrame({
        'Date': features['date'],
        'Latitude': numerical_features['latitude'],
        'Longitude': numerical_features['longitude'],
        'SWE_prediction': predictions_df['SWE_Predicted']
    })
    
    predictions_formatted_df.to_csv("swe_predictions.csv", index=False)
    print("SWE Predictions along with actual values saved to swe_predictions.csv")

    observed = predictions_df['SWE_Actual']
    predicted = predictions_df['SWE_Predicted']
    nse = nash_sutcliffe_efficiency(observed, predicted)
    rel_bias = relative_bias(observed, predicted)
    act_error = actual_error(observed, predicted)
    rmse = root_mean_square_error(observed, predicted)

    print(f"Nash Sutcliffe Efficiency (NSE): {nse}")
    print(f"Relative Bias (%): {rel_bias}")
    print(f"Actual Error: {act_error}")
    print(f"Root Mean Square Error (RMSE): {rmse}")
