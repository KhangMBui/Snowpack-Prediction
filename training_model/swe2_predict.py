import argparse
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMModel
from performance_metrics import nash_sutcliffe_efficiency, relative_bias, actual_error, root_mean_square_error

# Load pre-trained model function
def load_model():
    model = LSTMModel(input_size=10, hidden_size=229, num_layers=2, output_size=1, dropout=0.2819993210162027)  # These should match your trained model's architecture
    model.load_state_dict(torch.load("./cuda_lstm_model.pth"))
    model = model.to("cuda")
    return model

# Predict and evaluate
def predict_and_evaluate(model, data_path, scaler, numerical_features, target_col):
    # Load data
    data = pd.read_csv(data_path)
    X = data[numerical_features].values
    y = data[target_col].values

    # Scale data
    X = scaler.transform(X)
    y = scaler.transform(y.reshape(-1, 1))

    # Create DataLoader
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)

    # Predict
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to("cuda")
            preds = model(X_batch.unsqueeze(1)).cpu().numpy()
            predictions.extend(preds)
    
    predictions = scaler.inverse_transform(predictions)
    data['SWE_Predicted'] = predictions
    
    # Save predictions
    predictions_df = data[['date', 'latitude', 'longitude', 'SWE_Predicted']]
    predictions_df.to_csv("swe_predictions_output.csv", index=False)
    print("Predictions saved to swe_predictions_output.csv")
    
    # Evaluate metrics
    observed = data[target_col].values
    predicted = predictions.flatten()
    nse = nash_sutcliffe_efficiency(observed, predicted)
    rel_bias = relative_bias(observed, predicted)
    act_error = actual_error(observed, predicted)
    rmse = root_mean_square_error(observed, predicted)

    print(f"Nash Sutcliffe Efficiency (NSE): {nse}")
    print(f"Relative Bias (%): {rel_bias}")
    print(f"Actual Error: {act_error}")
    print(f"Root Mean Square Error (RMSE): {rmse}")

    # Plot results using Pandas Visualization
    data.set_index('date', inplace=True)
    data['SWE_Predicted'].plot(figsize=(10, 5), title='SWE Prediction')
    plt.ylabel('SWE Prediction')
    plt.show()

    # Plot results
    # plt.figure(figsize=(10, 5))
    # plt.plot(data['date'], observed, label='Actual SWE')
    # plt.plot(data['date'], predicted, label='Predicted SWE', linestyle='--')
    # plt.xlabel('Date')
    # plt.ylabel('SWE Prediction')
    # plt.title('Figure 1: SWE curve for a station (Green Lake) for the water year 1984')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

# Main function to parse arguments and run the program
def main():
    # Load pre-trained model
    model = load_model()

    # Load scaler and features from previous preprocessing (pseudo-code)
    scaler = MinMaxScaler()  # Replace with your actual scaler
    scaler.fit([[-1, 1]])  # Dummy fit for demonstration; replace with actual fitting process

    numerical_features = ['latitude','longitude','SWE','precip','tmin','tmax','SPH','SRAD','Rmax','Rmin','windspeed','Elevation','Southness']

    # Predict and evaluate
    predict_and_evaluate(model, "/scratch/project/hackathon/team6/test_data.csv", scaler, numerical_features, "SWE")

if __name__ == "__main__":
    main()
