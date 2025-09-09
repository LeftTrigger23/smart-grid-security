import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder component for the smart grid security model
    Reconstructs the original input from the latent representation
    """
    def __init__(self, latent_dim, output_shape):
        """
        Initialize the decoder
        
        Args:
            latent_dim: Dimension of the latent space
            output_shape: Tuple (time_steps, features)
        """
        super(Decoder, self).__init__()
        self.time_steps, self.features = output_shape
        
        # Calculate sizes for reconstruction
        self.time_steps_reduced = self.time_steps // 4
        
        # Dense layers to expand dimensions
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(64, self.time_steps_reduced * self.features),
            nn.ReLU()
        )
        
        # Transpose convolution layers
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        
        # Output convolution
        self.conv_out = nn.Conv1d(
            in_channels=32,
            out_channels=self.features,
            kernel_size=3,
            padding=1
        )
        
    def forward(self, x):
        """
        Forward pass through the decoder
        
        Args:
            x: Latent representation of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed data of shape (batch_size, time_steps, features)
        """
        batch_size = x.size(0)
        
        # Apply dense layers to expand dimensions
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Reshape for convolutional layers
        x = x.view(batch_size, self.features, self.time_steps_reduced)
        
        # Apply transpose convolution layers
        x = self.upconv1(x)
        x = self.upconv2(x)
        
        # Apply output convolution
        x = self.conv_out(x)
        
        # Transpose to original format (batch_size, time_steps, features)
        x = x.permute(0, 2, 1)
        
        return x

def build_decoder(latent_dim, output_shape):
    """
    Build the decoder component
    
    Args:
        latent_dim: Dimension of the latent space
        output_shape: Tuple (time_steps, features)
        
    Returns:
        PyTorch Decoder model
    """
    return Decoder(latent_dim, output_shape)
