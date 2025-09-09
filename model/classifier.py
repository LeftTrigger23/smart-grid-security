import torch
import torch.nn as nn

class Classifier(nn.Module):
    """
    Classifier component for identifying attack types
    Operates on the latent representations from the encoder
    """
    def __init__(self, latent_dim, num_classes, dropout_rate=0.2):
        """
        Initialize the classifier
        
        Args:
            latent_dim: Dimension of the latent space
            num_classes: Number of attack types to classify
            dropout_rate: Dropout rate for regularization
        """
        super(Classifier, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the classifier
        
        Args:
            x: Latent representation of shape (batch_size, latent_dim)
            
        Returns:
            Logits for each class
        """
        return self.model(x)

def build_classifier(latent_dim, num_classes, dropout_rate=0.2):
    """
    Build the classifier component
    
    Args:
        latent_dim: Dimension of the latent space
        num_classes: Number of attack types to classify
        dropout_rate: Dropout rate for regularization
        
    Returns:
        PyTorch Classifier model
    """
    return Classifier(latent_dim, num_classes, dropout_rate)
