"""
Graph Neural Network for Supply Chain Risk Analysis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import networkx as nx
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class SupplyChainGNN(nn.Module):
    """Graph Neural Network for supply chain risk analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 3, num_layers: int = 3):
        super(SupplyChainGNN, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index, batch=None):
        # Apply GCN layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer (no activation)
        x = self.convs[-1](x, edge_index)
        
        # Global pooling for graph-level prediction
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return F.log_softmax(x, dim=1)

class SupplyChainRiskAnalyzer:
    """Supply chain risk analysis using Graph Neural Networks"""
    
    def __init__(self, feature_dim: int = 10):
        self.feature_dim = feature_dim
        self.model = SupplyChainGNN(feature_dim)
        self.scaler = StandardScaler()
        self.risk_categories = ['LOW', 'MEDIUM', 'HIGH']
        self.is_trained = False
        self.graph_data = {}
        
    def create_business_graph(self, business_relationships: pd.DataFrame, 
                            business_features: pd.DataFrame) -> Data:
        """Create a graph from business relationships and features"""
        
        # Create node mapping
        all_businesses = pd.concat([
            business_relationships['supplier_id'], 
            business_relationships['buyer_id']
        ]).unique()
        
        node_mapping = {business_id: idx for idx, business_id in enumerate(all_businesses)}
        
        # Create edge list
        edge_list = []
        edge_weights = []
        
        for _, row in business_relationships.iterrows():
            supplier_idx = node_mapping[row['supplier_id']]
            buyer_idx = node_mapping[row['buyer_id']]
            
            edge_list.append([supplier_idx, buyer_idx])
            edge_list.append([buyer_idx, supplier_idx])  # Undirected graph
            
            # Edge weights based on transaction volume/importance
            weight = row.get('transaction_volume', 1.0)
            edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
        
        # Create node features
        node_features = []
        for business_id in all_businesses:
            if business_id in business_features.index:
                features = business_features.loc[business_id].values
            else:
                # Default features for unknown businesses
                features = np.zeros(self.feature_dim)
            
            node_features.append(features)
        
        node_features = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Create PyTorch Geometric data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        return graph_data, node_mapping
    
    def prepare_training_data(self, graphs_data: List[Tuple[Data, str]]) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders"""
        
        # Convert risk labels to integers
        label_mapping = {label: idx for idx, label in enumerate(self.risk_categories)}
        
        processed_graphs = []
        for graph_data, risk_label in graphs_data:
            graph_data.y = torch.tensor([label_mapping[risk_label]], dtype=torch.long)
            processed_graphs.append(graph_data)
        
        # Split into train/val
        split_idx = int(0.8 * len(processed_graphs))
        train_graphs = processed_graphs[:split_idx]
        val_graphs = processed_graphs[split_idx:]
        
        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """Train the GNN model"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                    total_val_loss += loss.item()
                    
                    pred = out.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = correct / total
            
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            if epoch % 20 == 0:
                logger.info(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, '
                          f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        self.is_trained = True
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    
    def predict_supply_chain_risk(self, graph_data: Data) -> Dict[str, Any]:
        """Predict supply chain risk for a business network"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with torch.no_grad():
            graph_data = graph_data.to(device)
            out = self.model(graph_data.x, graph_data.edge_index)
            
            # Graph-level prediction
            graph_pred = torch.mean(out, dim=0, keepdim=True)
            probabilities = F.softmax(graph_pred, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            
            # Node-level predictions (individual business risks)
            node_probabilities = F.softmax(out, dim=1)
            node_predictions = node_probabilities.argmax(dim=1)
        
        return {
            'overall_risk': self.risk_categories[predicted_class],
            'risk_probabilities': {
                self.risk_categories[i]: float(probabilities[0, i])
                for i in range(len(self.risk_categories))
            },
            'node_risks': [self.risk_categories[pred.item()] for pred in node_predictions],
            'node_risk_scores': node_probabilities.cpu().numpy()
        }
    
    def analyze_network_centrality(self, graph_data: Data, node_mapping: Dict) -> Dict[str, Any]:
        """Analyze network centrality and identify critical nodes"""
        
        # Convert to NetworkX for centrality analysis
        edge_index = graph_data.edge_index.cpu().numpy()
        G = nx.Graph()
        
        # Add edges
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        pagerank = nx.pagerank(G)
        
        # Identify critical nodes
        reverse_mapping = {v: k for k, v in node_mapping.items()}
        
        critical_nodes = []
        for node_idx in G.nodes():
            business_id = reverse_mapping.get(node_idx, f"node_{node_idx}")
            
            centrality_score = (
                degree_centrality[node_idx] * 0.3 +
                betweenness_centrality[node_idx] * 0.3 +
                closeness_centrality[node_idx] * 0.2 +
                pagerank[node_idx] * 0.2
            )
            
            critical_nodes.append({
                'business_id': business_id,
                'node_index': node_idx,
                'centrality_score': centrality_score,
                'degree_centrality': degree_centrality[node_idx],
                'betweenness_centrality': betweenness_centrality[node_idx],
                'closeness_centrality': closeness_centrality[node_idx],
                'pagerank': pagerank[node_idx]
            })
        
        # Sort by centrality score
        critical_nodes.sort(key=lambda x: x['centrality_score'], reverse=True)
        
        return {
            'critical_nodes': critical_nodes[:10],  # Top 10 critical nodes
            'network_stats': {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'density': nx.density(G),
                'avg_clustering': nx.average_clustering(G),
                'num_connected_components': nx.number_connected_components(G)
            }
        }