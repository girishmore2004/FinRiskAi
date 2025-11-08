"""
Federated Learning implementation for privacy-preserving credit scoring
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import logging

logger = logging.getLogger(__name__)

class FederatedCreditModel(nn.Module):
    """Neural network model for federated credit scoring"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32], dropout_rate: float = 0.3):
        super(FederatedCreditModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 2))  # Binary classification
        layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class FederatedLearningSystem:
    """Federated Learning coordinator for credit scoring"""
    
    def __init__(self, input_size: int, num_clients: int = 5):
        self.input_size = input_size
        self.num_clients = num_clients
        self.global_model = FederatedCreditModel(input_size)
        self.client_models = {}
        self.client_data = {}
        self.global_rounds = 0
        
        # Initialize client models
        for client_id in range(num_clients):
            self.client_models[client_id] = copy.deepcopy(self.global_model)
    
    def add_client_data(self, client_id: int, X: pd.DataFrame, y: pd.Series):
        """Add training data for a specific client (bank)"""
        if client_id not in range(self.num_clients):
            raise ValueError(f"Client ID must be between 0 and {self.num_clients-1}")
        
        self.client_data[client_id] = {
            'X': torch.FloatTensor(X.values),
            'y': torch.LongTensor(y.values)
        }
        logger.info(f"Added {len(X)} samples for client {client_id}")
    
    def local_train(self, client_id: int, epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001) -> Dict[str, float]:
        """Train model locally on client data"""
        if client_id not in self.client_data:
            raise ValueError(f"No data available for client {client_id}")
        
        model = self.client_models[client_id]
        data = self.client_data[client_id]
        
        # Create DataLoader
        dataset = TensorDataset(data['X'], data['y'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / num_batches
        logger.info(f"Client {client_id} local training completed. Average loss: {avg_loss:.4f}")
        
        return {'loss': avg_loss, 'samples': len(data['X'])}
    
    def federated_averaging(self, client_weights: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """Perform federated averaging (FedAvg) to update global model"""
        if client_weights is None:
            # Equal weighting
            client_weights = {i: 1.0/len(self.client_models) for i in self.client_models.keys()}
        else:
            # Normalize weights
            total_weight = sum(client_weights.values())
            client_weights = {k: v/total_weight for k, v in client_weights.items()}
        
        # Get global model state dict
        global_state_dict = self.global_model.state_dict()
        
        # Initialize averaged parameters
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.zeros_like(global_state_dict[key])
        
        # Weighted average of client model parameters
        for client_id, weight in client_weights.items():
            client_state_dict = self.client_models[client_id].state_dict()
            for key in global_state_dict.keys():
                global_state_dict[key] += weight * client_state_dict[key]
        
        # Update global model
        self.global_model.load_state_dict(global_state_dict)
        
        # Update all client models with new global model
        for client_id in self.client_models.keys():
            self.client_models[client_id].load_state_dict(global_state_dict)
        
        self.global_rounds += 1
        
        return {
            'global_round': self.global_rounds,
            'client_weights': client_weights,
            'participating_clients': list(client_weights.keys())
        }
    
    def federated_train(self, num_rounds: int = 10, local_epochs: int = 5, 
                       participation_rate: float = 1.0) -> List[Dict[str, Any]]:
        """Complete federated training process"""
        training_history = []
        
        for round_num in range(num_rounds):
            logger.info(f"Starting federated round {round_num + 1}/{num_rounds}")
            
            # Select participating clients
            available_clients = list(self.client_data.keys())
            num_participants = max(1, int(len(available_clients) * participation_rate))
            participating_clients = np.random.choice(available_clients, num_participants, replace=False)
            
            # Local training
            client_results = {}
            for client_id in participating_clients:
                result = self.local_train(client_id, epochs=local_epochs)
                client_results[client_id] = result
            
            # Calculate client weights based on data size
            client_weights = {}
            total_samples = sum([r['samples'] for r in client_results.values()])
            for client_id, result in client_results.items():
                client_weights[client_id] = result['samples'] / total_samples
            
            # Federated averaging
            fed_result = self.federated_averaging(client_weights)
            
            # Record training round results
            round_result = {
                'round': round_num + 1,
                'participating_clients': list(participating_clients),
                'client_losses': {k: v['loss'] for k, v in client_results.items()},
                'average_loss': np.mean([v['loss'] for v in client_results.values()]),
                'total_samples': total_samples
            }
            training_history.append(round_result)
            
            logger.info(f"Round {round_num + 1} completed. Average loss: {round_result['average_loss']:.4f}")
        
        return training_history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the global model"""
        self.global_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            outputs = self.global_model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).numpy()
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using the global model"""
        self.global_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            outputs = self.global_model(X_tensor)
            probabilities = outputs.numpy()
        return probabilities
    
    def evaluate_global_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the global model on test data"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted'),
            'auc_roc': roc_auc_score(y_test, probabilities[:, 1])
        }
        
        return metrics