'''
collision_model.py
This module defines a neural network model for predicting collision probability 
and risk classification between satellites.
Also defines a dataset class for loading and processing the data.
'''
import torch
import torch.nn as nn

class CollisionRiskModel(nn.Module):
    """
    Multi-task model for collision probability prediction (log-space) and risk classification
    """
    def __init__(self, input_dim=7, hidden_dim=128, num_classes=3):
        super(CollisionRiskModel, self).__init__()
        
        # Shared feature extractor (same as before)
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )
        
        # Log-probability regression head
        self.log_prob_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
            # No activation - output raw log probabilities
        )
        
        # Risk classification head (same as before)
        self.class_head = nn.Linear(hidden_dim // 4, num_classes)
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        
        # Predict log10 probability (typically in range [-12, -3])
        log_prob_pred = self.log_prob_head(shared_features)
        # Soft clamping (smoother):
        log_prob_pred = 3.5 * torch.tanh(log_prob_pred / 3.5) - 6.5
        
        # Convert back to linear space for evaluation
        prob_pred = torch.pow(10, log_prob_pred)
        
        # Classification prediction
        class_pred = self.class_head(shared_features)
        
        return prob_pred.squeeze(), class_pred, log_prob_pred.squeeze()

# Update dataset class to handle log probabilities
class CollisionDataset(torch.utils.data.Dataset):
    """
    Dataset for multi-task collision risk learning with log-space probabilities
    """
    def __init__(self, features, log_collision_probs, risk_classes):
        self.features = torch.FloatTensor(features)
        self.log_collision_probs = torch.FloatTensor(log_collision_probs)
        self.risk_classes = torch.LongTensor(risk_classes)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (self.features[idx], 
                self.log_collision_probs[idx], 
                self.risk_classes[idx])