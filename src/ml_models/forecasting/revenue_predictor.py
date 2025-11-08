"""
Advanced time series forecasting for SME revenue prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from fbprophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class LSTMRevenuePredictor(nn.Module):
    """LSTM model for revenue forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int = 50, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMRevenuePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        predictions = self.linear(lstm_out[:, -1, :])  # Use last time step
        return predictions

class HybridRevenueForecaster:
    """Hybrid forecasting model combining LSTM and Prophet"""
    
    def __init__(self, sequence_length: int = 12):
        self.sequence_length = sequence_length
        self.lstm_model = None
        self.prophet_model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def prepare_lstm_data(self, data: pd.DataFrame, target_col: str = 'revenue') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data[[target_col]])
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def train_lstm(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train LSTM model"""
        # Reshape for LSTM input
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.lstm_model = LSTMRevenuePredictor(input_size=1, hidden_size=50, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        
        # Training loop
        self.lstm_model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.lstm_model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f'LSTM Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.6f}')
    
    def train_prophet(self, data: pd.DataFrame, date_col: str = 'date', target_col: str = 'revenue'):
        """Train Prophet model"""
        prophet_data = data[[date_col, target_col]].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Add external regressors if available
        if 'economic_indicator' in data.columns:
            prophet_data['economic_indicator'] = data['economic_indicator']
            self.prophet_model.add_regressor('economic_indicator')
        
        self.prophet_model.fit(prophet_data)
    
    def train(self, data: pd.DataFrame, date_col: str = 'date', target_col: str = 'revenue') -> Dict[str, Any]:
        """Train both LSTM and Prophet models"""
        print("Training Hybrid Revenue Forecaster...")
        
        # Prepare data
        data = data.sort_values(date_col).reset_index(drop=True)
        
        # Train LSTM
        print("Training LSTM component...")
        X_lstm, y_lstm = self.prepare_lstm_data(data, target_col)
        self.train_lstm(X_lstm, y_lstm)
        
        # Train Prophet
        print("Training Prophet component...")
        self.train_prophet(data, date_col, target_col)
        
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.predict(data, periods=0)  # In-sample predictions
        actual_values = data[target_col].values[-len(train_predictions):]
        
        mae = mean_absolute_error(actual_values, train_predictions)
        rmse = np.sqrt(mean_squared_error(actual_values, train_predictions))
        mape = np.mean(np.abs((actual_values - train_predictions) / actual_values)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'model_components': ['LSTM', 'Prophet']
        }
    
    def predict_lstm(self, last_sequence: np.ndarray, periods: int) -> np.ndarray:
        """Generate LSTM predictions"""
        self.lstm_model.eval()
        predictions = []
        
        current_sequence = last_sequence.copy()
        
        with torch.no_grad():
            for _ in range(periods):
                # Reshape for model input
                input_tensor = torch.FloatTensor(current_sequence).view(1, -1, 1)
                
                # Make prediction
                pred = self.lstm_model(input_tensor)
                pred_value = pred.item()
                predictions.append(pred_value)
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[1:], pred_value)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions
    
    def predict_prophet(self, periods: int, freq: str = 'M') -> np.ndarray:
        """Generate Prophet predictions"""
        # Create future dates
        future = self.prophet_model.make_future_dataframe(periods=periods, freq=freq)
        
        # Make predictions
        forecast = self.prophet_model.predict(future)
        
        # Return only the forecasted values
        return forecast['yhat'].values[-periods:]
    
    def predict(self, historical_data: pd.DataFrame, periods: int = 12, lstm_weight: float = 0.6) -> np.ndarray:
        """Generate ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # LSTM predictions
        last_sequence = self.scaler.transform(historical_data[['revenue']].tail(self.sequence_length)).flatten()
        lstm_predictions = self.predict_lstm(last_sequence, periods)
        
        # Prophet predictions
        prophet_predictions = self.predict_prophet(periods)
        
        # Ensure same length
        min_length = min(len(lstm_predictions), len(prophet_predictions))
        lstm_predictions = lstm_predictions[:min_length]
        prophet_predictions = prophet_predictions[:min_length]
        
        # Ensemble prediction (weighted average)
        ensemble_predictions = lstm_weight * lstm_predictions + (1 - lstm_weight) * prophet_predictions
        
        return ensemble_predictions
    
    def calculate_confidence_intervals(self, predictions: np.ndarray, confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        # Simple approach using historical volatility
        std_multiplier = 1.96 if confidence_level == 0.95 else 2.58  # 99% confidence
        
        # Estimate volatility from predictions (simplified)
        volatility = np.std(predictions) * 0.1  # 10% of prediction std as uncertainty
        
        lower_bound = predictions - std_multiplier * volatility
        upper_bound = predictions + std_multiplier * volatility
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }
    
    def assess_forecast_risk(self, predictions: np.ndarray, current_revenue: float) -> Dict[str, Any]:
        """Assess risk based on revenue forecasts"""
        avg_predicted_revenue = np.mean(predictions)
        revenue_trend = (predictions[-1] - predictions[0]) / len(predictions)
        volatility = np.std(predictions) / np.mean(predictions)
        
        # Risk assessment
        if avg_predicted_revenue < current_revenue * 0.8:
            risk_level = 'HIGH'
        elif avg_predicted_revenue < current_revenue * 0.9:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_level': risk_level,
            'average_predicted_revenue': avg_predicted_revenue,
            'revenue_trend': revenue_trend,
            'volatility': volatility,
            'growth_rate': (avg_predicted_revenue - current_revenue) / current_revenue
        }
