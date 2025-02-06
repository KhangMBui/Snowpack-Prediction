import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import MinMaxScaler
from performance_metrics import nash_sutcliffe_efficiency, relative_bias, actual_error, root_mean_square_error
import pandas as pd

def train_model(model, X_train, y_train, X_test, y_test, scaler, features, numerical_features, target, best_params, num_epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    grad_scaler = GradScaler()
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    accumulation_steps = 4  # Number of mini-batches to accumulate
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to("cuda")
            y_batch = y_batch.to("cuda")
            
            with autocast():
                outputs = model(X_batch.unsqueeze(1))
                loss = criterion(outputs, y_batch)
            grad_scaler.scale(loss).backward()
            
            if (i+1) % accumulation_steps == 0:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    
    # Use sklearn's MinMaxScaler for scaling target values
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(target.values.reshape(-1, 1))
    
    X_test = X_test.to("cuda")  # Move X_test to the same device
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.unsqueeze(1)).cpu().numpy()
        predictions_original_scale = min_max_scaler.inverse_transform(predictions)
    
    predictions_df = pd.DataFrame(X_test.cpu().numpy(), columns=numerical_features.columns)
    predictions_df['SWE_Actual'] = y_test.cpu().numpy()
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

