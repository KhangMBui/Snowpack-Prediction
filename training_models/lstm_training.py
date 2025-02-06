import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from data_processing import load_and_preprocess_data
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model import LSTMModel
from train import train_model

def objective(trial):
    X_train, X_test, y_train, y_test, scaler, features, numerical_features, target = load_and_preprocess_data("/scratch/project/hackathon/team6/merged_data_location_sorted.csv")
    
    hidden_size = trial.suggest_int('hidden_size', 100, 500)  # Increased hidden size range
    num_layers = trial.suggest_int('num_layers', 2, 5)  # Increased number of layers
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    batch_size = trial.suggest_int('batch_size', 2048, 4096)  # Increased batch size range
    
    model = LSTMModel(input_size=X_train.shape[1], hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout)
    
    model = model.to("cuda")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to("cuda")
            y_batch = y_batch.to("cuda")
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

        # Ensure predictions have the same shape as the original scaled data
        predictions_original_scale = scaler.inverse_transform(predictions)
    
    mse = mean_squared_error(y_test.to("cuda"), predictions_original_scale)
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
