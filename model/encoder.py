import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder component for the smart grid security model
    Combines CNN and LSTM layers to extract spatial-temporal features
    """
    def __init__(self, input_shape, latent_dim):
        """
        Initialize the encoder
        
        Args:
            input_shape: Tuple (time_steps, features)
            latent_dim: Dimension of the latent space
        """
        super(Encoder, self).__init__()
        self.time_steps, self.features = input_shape
        
        # CNN layers for spatial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate the size after CNN layers
        cnn_output_size = (self.time_steps // 4) * 64
        
        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        # Dense layers for latent representation
        self.fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.fc2 = nn.Linear(64, latent_dim)
        
    def forward(self, x):
        """
        Forward pass through the encoder
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, features)
            
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        batch_size = x.size(0)
        
        # Transpose for CNN (batch_size, features, time_steps)
        x = x.permute(0, 2, 1)
        
        # Apply CNN layers
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Transpose back for LSTM (batch_size, time_steps, channels)
        x = x.permute(0, 2, 1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dense layers
        x = self.fc1(lstm_out)
        encoded = self.fc2(x)
        
        return encoded

def build_encoder(input_shape, latent_dim):
    """
    Build the encoder component
    
    Args:
        input_shape: Tuple (time_steps, features)
        latent_dim: Dimension of the latent space
        
    Returns:
        PyTorch Encoder model
    """
    return Encoder(input_shape, latent_dim)
