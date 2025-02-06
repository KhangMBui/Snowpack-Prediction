import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_processing import load_and_preprocess_data
from model import LSTMModel
from train import train_model

def objective(trial):
    X_train, X_test, y_train, y_test, scaler, features, numerical_features, target = load_and_preprocess_data("/scratch/project/hackathon/team6/merged_data_location_sorted.csv")
    
    hidden_size = trial.suggest_int('hidden_size', 100, 500)  # Increased hidden size range
    num_layers = trial.suggest_int('num_layers', 2, 5)  # Increased number of layers
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    batch_size = trial.suggest_int('batch_size', 4096, 8192)  # Increased batch size range
    
    model = LSTMModel(input_size=X_train.shape[1], hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to("cuda")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    grad_scaler = GradScaler()
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    num_epochs = 50
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

        # Ensure predictions have the same shape as the original scaled data
        predictions_original_scale = min_max_scaler.inverse_transform(predictions)
    
    mse = mean_squared_error(y_test.cpu(), predictions_original_scale)
    return mse

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    best_params = study.best_params
    X_train, X_test, y_train, y_test, scaler, features, numerical_features, target = load_and_preprocess_data("/scratch/project/hackathon/team6/merged_data_location_sorted.csv")
    
    model = LSTMModel(input_size=X_train.shape[1], hidden_size=best_params['hidden_size'],
                      num_layers=best_params['num_layers'], output_size=1,
                      dropout=best_params['dropout'])
    
    train_model(model, X_train, y_train, X_test, y_test, scaler, features, numerical_features, target, best_params)

    # Save the model
    torch.save(model.state_dict(), 'lstm_model.pth')

if __name__ == "__main__":
    main()
