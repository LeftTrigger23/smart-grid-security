import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator component for the GAN
    Determines whether a latent representation is real or fake
    """
    def __init__(self, latent_dim, dropout_rate=0.2):
        """
        Initialize the discriminator
        
        Args:
            latent_dim: Dimension of the latent space
            dropout_rate: Dropout rate for regularization
        """
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through the discriminator
        
        Args:
            x: Latent representation of shape (batch_size, latent_dim)
            
        Returns:
            Probability that the input is real (from the encoder)
        """
        return self.model(x)

def build_discriminator(latent_dim, dropout_rate=0.2):
    """
    Build the discriminator component
    
    Args:
        latent_dim: Dimension of the latent space
        dropout_rate: Dropout rate for regularization
        
    Returns:
        PyTorch Discriminator model
    """
    return Discriminator(latent_dim, dropout_rate)
