import torch
import torch.nn as nn

class DiabetesANN(nn.Module):
    """
    Artificial Neural Network for binary classification of diabetes risk.
    Features:
    - Input Layer: 8 clinical features
    - Hidden Layer 1: 32 neurons with LeakyReLU activation and Batch Normalization
    - Hidden Layer 2: 16 neurons with LeakyReLU activation, Batch Normalization, and Dropout (0.2)
    - Output Layer: 1 neuron (logit output for BCEWithLogitsLoss)
    """
    def __init__(self, input_dim: int = 8, hidden_dim1: int = 32, hidden_dim2: int = 16):
        super().__init__()
        
        # Hidden Layer 1
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.act1 = nn.LeakyReLU(negative_slope=0.1)
        
        # Hidden Layer 2
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.act2 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(p=0.2)
        
        # Output Layer
        self.out = nn.Linear(hidden_dim2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Returns logits (pre-activation values).
        """
        # Pass through first layer
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Pass through second layer
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout(x)
        
        # Pass through output layer
        x = self.out(x)
        return x
