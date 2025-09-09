import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import argparse
import os

from model.encoder import build_encoder
from model.decoder import build_decoder
from model.discriminator import build_discriminator
from model.classifier import build_classifier
from model.autoencoder_gan import SmartGridSecurityModel
from utils.data_loader import load_data, preprocess_data

def create_improved_model_comparison(results_dir, hybrid_accuracy, cnn_accuracy, lstm_accuracy):
    """
    Create an improved model comparison visualization with dynamic scaling and 
    better visual differentiation between similar values
    """
    models = ['Hybrid', 'CNN', 'LSTM']
    accuracies = [hybrid_accuracy, cnn_accuracy, lstm_accuracy]
    colors = ['#3366CC', '#109618', '#FF9900']
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Determine if we need to adjust the scale
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    
    # Calculate the range and adjust the zoom level
    # Create more space between CNN and LSTM by narrowing the y-axis range
    # This makes small differences more visible
    y_min = max(0, min_acc - 0.003)  # Adjust this buffer value as needed
    y_max = min(1.0, max_acc + 0.003)  # Adjust this buffer value as needed
    
    # If the difference between CNN and LSTM is very small, zoom in even more
    if abs(cnn_accuracy - lstm_accuracy) < 0.005:
        # Find the minimum of the two
        min_of_two = min(cnn_accuracy, lstm_accuracy)
        # Set a smaller range to exaggerate the difference
        y_min = min_of_two - 0.003
        y_max = max(cnn_accuracy, lstm_accuracy) + 0.003
    
    plt.ylim(y_min, y_max)
    
    # Add note about zoomed axis
    plt.figtext(0.5, 0.01, 
              f"Note: Y-axis zoomed to range [{y_min:.3f}, {y_max:.3f}] to highlight differences",
              ha='center', fontsize=10, style='italic')
    
    # Create bar chart
    bars = plt.bar(models, accuracies, color=colors, width=0.6)
    
    # Add horizontal grid lines (more of them for better visual reference)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        # Position the label just above the bar
        label_y_position = height + (plt.ylim()[1] - plt.ylim()[0]) * 0.01
        plt.text(bar.get_x() + bar.get_width()/2., label_y_position,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Customize chart appearance
    plt.title('Model Accuracy Comparison', fontsize=16, pad=20)
    plt.ylabel('Accuracy', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Add more y-axis ticks for better reference
    plt.locator_params(axis='y', nbins=10)
    
    # Save the chart with tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for note if present
    plt.savefig(f"{results_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    
    # Create a second visualization: Error rates to highlight differences
    plt.figure(figsize=(12, 8))
    
    # Error rates (1 - accuracy) often show differences more clearly
    error_rates = [1 - acc for acc in accuracies]
    
    # Scale is much more visible with error rates
    bars = plt.bar(models, error_rates, color=colors, width=0.6)
    
    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        # Format as percentage for easier interpretation
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{height*100:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Customize appearance
    plt.title('Model Error Rates (Lower is Better)', fontsize=16, pad=20)
    plt.ylabel('Error Rate (1 - Accuracy)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(f"{results_dir}/error_rate_comparison.png", dpi=300, bbox_inches='tight')
    
    # Create a third visualization: Horizontal comparison
    plt.figure(figsize=(12, 8))
    
    # Horizontal bar chart often makes small differences more noticeable
    y_pos = np.arange(len(models))
    bars = plt.barh(y_pos, accuracies, color=colors, alpha=0.8)
    plt.yticks(y_pos, models, fontsize=14)
    
    # Set x-axis to highlight differences
    plt.xlim(y_min, y_max)
    
    # Add value labels inside each bar
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width - (y_max-y_min)*0.05, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='right', va='center', fontweight='bold', color='white')
    
    # Customize appearance
    plt.title('Model Accuracy Comparison (Horizontal)', fontsize=16)
    plt.xlabel('Accuracy (Zoomed Scale)', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add more x-axis ticks for better reference
    plt.locator_params(axis='x', nbins=10)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(f"{results_dir}/horizontal_comparison.png", dpi=300, bbox_inches='tight')
    
    print(f"Enhanced model comparison charts saved to {results_dir}/")

def parse_args():
    parser = argparse.ArgumentParser(description='Test individual components of the hybrid model')
    parser.add_argument('--data_path', type=str, default='./dataset/data/smart_grid_data.csv', help='Path to data file')
    parser.add_argument('--component', type=str, required=True, 
                        choices=['autoencoder', 'gan', 'cnn', 'lstm', 'classifier', 'all'],
                        help='Component to test')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--results_dir', type=str, default='component_tests', help='Directory to save results')
    return parser.parse_args()

def test_autoencoder(X_train, X_val, X_test, args, device):
    print("Testing AutoEncoder component...")
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    latent_dim = 32
    
    # Build encoder and decoder
    encoder = build_encoder(input_shape, latent_dim).to(device)
    decoder = build_decoder(latent_dim, input_shape).to(device)
    
    # Define optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Convert data to PyTorch tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        
    # Train autoencoder
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        
        # Shuffle training data
        indices = torch.randperm(len(X_train))
        
        # Training loop
        train_loss = 0.0
        for i in range(0, len(X_train), args.batch_size):
            # Get batch
            batch_indices = indices[i:i+args.batch_size]
            batch_x = X_train[batch_indices]
            
            # Zero gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            # Forward pass
            encoded = encoder(batch_x)
            decoded = decoder(encoded)
            
            # Calculate loss
            loss = criterion(decoded, batch_x)
            
            # Backward pass and optimize
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            # Update loss
            train_loss += loss.item() * len(batch_x)
        
        train_loss /= len(X_train)
        train_losses.append(train_loss)
        
        # Validation
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            encoded_val = encoder(X_val)
            decoded_val = decoder(encoded_val)
            val_loss = criterion(decoded_val, X_val).item()
            val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Evaluate on test set
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        encoded_test = encoder(X_test)
        decoded_test = decoder(encoded_test)
        test_loss = criterion(decoded_test, X_test).item()
        print(f"Test reconstruction loss (MSE): {test_loss:.6f}")
    
    # Calculate reconstruction error
    with torch.no_grad():
        mse = torch.mean(torch.square(X_test - decoded_test), dim=(1, 2)).cpu().numpy()
        threshold = np.mean(mse) + 3 * np.std(mse)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot training history
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('AutoEncoder Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    
    # Plot sample reconstructions
    n_samples = 5
    samples = X_test[:n_samples].cpu().numpy()
    reconstructed = decoded_test[:n_samples].cpu().numpy()
    
    for i in range(n_samples):
        # Original
        plt.subplot(n_samples, 2, 2*i + 1)
        plt.plot(samples[i, :, 0])
        if i == 0:
            plt.title('Original')
        
        # Reconstructed
        plt.subplot(n_samples, 2, 2*i + 2)
        plt.plot(reconstructed[i, :, 0])
        if i == 0:
            plt.title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/autoencoder_results.png")
    
    return {
        'loss': test_loss,
        'mse': mse,
        'threshold': threshold,
        'encoder': encoder,
        'decoder': decoder
    }
'''
def test_gan(X_train, X_val, X_test, args, device):
    print("Testing GAN component...")
    
    # Create models
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    latent_dim = 32
    
    # Build encoder and discriminator
    encoder = build_encoder(input_shape, latent_dim).to(device)
    discriminator = build_discriminator(latent_dim).to(device)
    
    # Define optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    
    # Define loss function
    criterion = nn.BCELoss()
    
    # Convert data to PyTorch tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Training GAN
    d_losses = []
    g_losses = []
    
    for epoch in range(args.epochs):
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), args.batch_size):
            # Get batch
            batch_indices = indices[i:i+args.batch_size]
            batch_x = X_train[batch_indices]
            batch_size_actual = len(batch_x)
            
            # Create labels
            real_labels = torch.ones(batch_size_actual, 1, device=device)
            fake_labels = torch.zeros(batch_size_actual, 1, device=device)
            
            # -----------------
            # Train Discriminator
            # -----------------
            discriminator_optimizer.zero_grad()
            
            # Real samples
            with torch.no_grad():
                real_encoded = encoder(batch_x)
            real_outputs = discriminator(real_encoded.detach())
            d_loss_real = criterion(real_outputs, real_labels)
            d_loss_real.backward()
            
            # Fake samples (noise)
            noise = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_outputs = discriminator(noise)
            d_loss_fake = criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()
            
            discriminator_optimizer.step()
            
            d_loss = (d_loss_real.item() + d_loss_fake.item()) / 2
            d_loss_epoch += d_loss * batch_size_actual
            
            # -----------------
            # Train Generator (Encoder)
            # -----------------
            encoder_optimizer.zero_grad()
            
            # Get latent representations
            encoded = encoder(batch_x)
            validity = discriminator(encoded)
            
            # Train encoder to fool discriminator
            g_loss = criterion(validity, real_labels)
            g_loss.backward()
            encoder_optimizer.step()
            
            g_loss_epoch += g_loss.item() * batch_size_actual
        
        # Calculate average losses
        d_loss_epoch /= len(X_train)
        g_loss_epoch /= len(X_train)
        
        d_losses.append(d_loss_epoch)
        g_losses.append(g_loss_epoch)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} [D loss: {d_loss_epoch:.4f}] [G loss: {g_loss_epoch:.4f}]")
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.title('GAN Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{args.results_dir}/gan_loss.png")
    
    # Generate latent representations for validation data
    encoder.eval()
    with torch.no_grad():
        encoded_val = encoder(X_val)
        encoded_val = encoded_val.cpu().numpy()
    
    # Visualize latent space
    plt.figure(figsize=(10, 8))
    plt.scatter(encoded_val[:, 0], encoded_val[:, 1], alpha=0.5)
    plt.title('GAN Latent Space (First 2 Dimensions)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(f"{args.results_dir}/gan_latent_space.png")
    
    return {
        'd_loss': d_losses[-1],
        'g_loss': g_losses[-1],
        'encoder': encoder,
        'discriminator': discriminator
    }
'''
def test_gan(X_train, X_val, X_test, args, device):
    print("Testing GAN component...")
    
    # Create models
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    latent_dim = 32
    
    # Build encoder and discriminator
    encoder = build_encoder(input_shape, latent_dim).to(device)
    discriminator = build_discriminator(latent_dim).to(device)
    
    # Define optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    
    # Define loss function
    criterion = nn.BCELoss()
    
    # Convert data to PyTorch tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Training GAN
    d_losses = []
    g_losses = []
    
    for epoch in range(args.epochs):
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), args.batch_size):
            # Get batch
            batch_indices = indices[i:i+args.batch_size]
            batch_x = X_train[batch_indices]
            batch_size_actual = len(batch_x)
            
            # Create labels
            real_labels = torch.ones(batch_size_actual, 1, device=device)
            fake_labels = torch.zeros(batch_size_actual, 1, device=device)
            
            # -----------------
            # Train Discriminator
            # -----------------
            discriminator_optimizer.zero_grad()
            
            # Real samples
            with torch.no_grad():
                real_encoded = encoder(batch_x)
            real_outputs = discriminator(real_encoded.detach())
            d_loss_real = criterion(real_outputs, real_labels)
            d_loss_real.backward()
            
            # Fake samples (noise)
            noise = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_outputs = discriminator(noise)
            d_loss_fake = criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()
            
            discriminator_optimizer.step()
            
            d_loss = (d_loss_real.item() + d_loss_fake.item()) / 2
            d_loss_epoch += d_loss * batch_size_actual
            
            # -----------------
            # Train Generator (Encoder)
            # -----------------
            encoder_optimizer.zero_grad()
            
            # Get latent representations
            encoded = encoder(batch_x)
            validity = discriminator(encoded)
            
            # Train encoder to fool discriminator
            g_loss = criterion(validity, real_labels)
            g_loss.backward()
            encoder_optimizer.step()
            
            g_loss_epoch += g_loss.item() * batch_size_actual
        
        # Calculate average losses
        d_loss_epoch /= len(X_train)
        g_loss_epoch /= len(X_train)
        
        d_losses.append(d_loss_epoch)
        g_losses.append(g_loss_epoch)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} [D loss: {d_loss_epoch:.4f}] [G loss: {g_loss_epoch:.4f}]")
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.title('GAN Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{args.results_dir}/gan_loss.png")
    
    # Generate latent representations for validation data
    encoder.eval()
    with torch.no_grad():
        encoded_val = encoder(X_val)
        encoded_val = encoded_val.cpu().numpy()
    
    # Visualize latent space
    plt.figure(figsize=(10, 8))
    plt.scatter(encoded_val[:, 0], encoded_val[:, 1], alpha=0.5)
    plt.title('GAN Latent Space (First 2 Dimensions)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(f"{args.results_dir}/gan_latent_space.png")
    
    # Add t-SNE visualization
    try:
        from sklearn.manifold import TSNE
        
        # Apply t-SNE for dimensionality reduction
        print("Generating t-SNE visualization of latent space...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, 
                    learning_rate='auto', init='pca')
        encoded_tsne = tsne.fit_transform(encoded_val)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot
        plt.scatter(encoded_tsne[:, 0], encoded_tsne[:, 1], alpha=0.7, s=50)
        
        # Add title and labels
        plt.title('t-SNE Visualization of GAN Latent Space (32D → 2D)', fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Improve aesthetics
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{args.results_dir}/gan_latent_space_tsne.png", dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved as '{args.results_dir}/gan_latent_space_tsne.png'")
    except Exception as e:
        print(f"Note: Could not create t-SNE visualization: {e}")
    
    return {
        'd_loss': d_losses[-1],
        'g_loss': g_losses[-1],
        'encoder': encoder,
        'discriminator': discriminator
    }

def test_cnn(X_train, X_val, X_test, y_train, y_val, y_test, args, device):
    print("Testing CNN component...")
    
    # Convert data to PyTorch tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Convert one-hot encoded targets to class indices
    if y_train.dim() > 1 and y_train.shape[1] > 1:
        y_train_cls = torch.argmax(y_train, dim=1)
        y_val_cls = torch.argmax(y_val, dim=1)
        y_test_cls = torch.argmax(y_test, dim=1)
    else:
        y_train_cls = y_train.long()
        y_val_cls = y_val.long()
        y_test_cls = y_test.long()
    
    # Define CNN model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    num_classes = y_train.shape[1] if y_train.dim() > 1 else 1
    
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            
            # CNN layers
            self.cnn_layers = nn.Sequential(
                nn.Conv1d(input_shape[1], 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()
            )
            
            # Calculate output size after CNN layers
            cnn_output_size = 64 * (input_shape[0] // 4)
            
            # Dense layers
            self.fc_layers = nn.Sequential(
                nn.Linear(cnn_output_size, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            # Transpose for CNN (batch_size, features, time_steps)
            x = x.permute(0, 2, 1)
            x = self.cnn_layers(x)
            x = self.fc_layers(x)
            return x
    
    # Create model and move to device
    cnn_model = CNNModel().to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(args.epochs):
        cnn_model.train()
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        
        train_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(0, len(X_train), args.batch_size):
            # Get batch
            batch_indices = indices[i:i+args.batch_size]
            batch_x = X_train[batch_indices]
            batch_y = y_train_cls[batch_indices]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = cnn_model(batch_x)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * len(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(X_train)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        cnn_model.eval()
        with torch.no_grad():
            outputs_val = cnn_model(X_val)
            val_loss = criterion(outputs_val, y_val_cls).item()
            _, predicted_val = torch.max(outputs_val.data, 1)
            val_acc = (predicted_val == y_val_cls).sum().item() / len(y_val_cls)
            
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Evaluate on test set
    cnn_model.eval()
    with torch.no_grad():
        outputs_test = cnn_model(X_test)
        test_loss = criterion(outputs_test, y_test_cls).item()
        _, predicted_test = torch.max(outputs_test.data, 1)
        test_acc = (predicted_test == y_test_cls).sum().item() / len(y_test_cls)
        
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Get predictions and convert to numpy for evaluation
    y_pred = predicted_test.cpu().numpy()
    y_true = y_test_cls.cpu().numpy()
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('CNN Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('CNN Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/cnn_performance.png")
    
    return {
        'accuracy': test_acc,
        'y_pred': y_pred,
        'y_true': y_true,
        'model': cnn_model
    }

def test_lstm(X_train, X_val, X_test, y_train, y_val, y_test, args, device):
    print("Testing LSTM component...")
    
    # Convert data to PyTorch tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Convert one-hot encoded targets to class indices
    if y_train.dim() > 1 and y_train.shape[1] > 1:
        y_train_cls = torch.argmax(y_train, dim=1)
        y_val_cls = torch.argmax(y_val, dim=1)
        y_test_cls = torch.argmax(y_test, dim=1)
    else:
        y_train_cls = y_train.long()
        y_val_cls = y_val.long()
        y_test_cls = y_test.long()
    
    # Define LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    num_classes = y_train.shape[1] if y_train.dim() > 1 else 1
    
    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            
            # LSTM layers
            self.lstm1 = nn.LSTM(
                input_size=input_shape[1],
                hidden_size=64,
                num_layers=1,
                batch_first=True
            )
            
            self.lstm2 = nn.LSTM(
                input_size=64,
                hidden_size=64,
                num_layers=1,
                batch_first=True
            )
            
            # Dense layers
            self.fc = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
        
        def forward(self, x):
            # First LSTM layer
            lstm1_out, _ = self.lstm1(x)
            
            # Second LSTM layer
            lstm2_out, _ = self.lstm2(lstm1_out)

            # Use the last time step from the output
            last_output = lstm2_out[:, -1, :]
            
            # Output layer
            x = self.fc(last_output)
            return x
    
    # Create model and move to device
    lstm_model = LSTMModel().to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(args.epochs):
        lstm_model.train()
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        
        train_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(0, len(X_train), args.batch_size):
            # Get batch
            batch_indices = indices[i:i+args.batch_size]
            batch_x = X_train[batch_indices]
            batch_y = y_train_cls[batch_indices]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = lstm_model(batch_x)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * len(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(X_train)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        lstm_model.eval()
        with torch.no_grad():
            outputs_val = lstm_model(X_val)
            val_loss = criterion(outputs_val, y_val_cls).item()
            _, predicted_val = torch.max(outputs_val.data, 1)
            val_acc = (predicted_val == y_val_cls).sum().item() / len(y_val_cls)
            
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Evaluate on test set
    lstm_model.eval()
    with torch.no_grad():
        outputs_test = lstm_model(X_test)
        test_loss = criterion(outputs_test, y_test_cls).item()
        _, predicted_test = torch.max(outputs_test.data, 1)
        test_acc = (predicted_test == y_test_cls).sum().item() / len(y_test_cls)
        
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Get predictions and convert to numpy for evaluation
    y_pred = predicted_test.cpu().numpy()
    y_true = y_test_cls.cpu().numpy()
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('LSTM Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('LSTM Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/lstm_performance.png")
    
    return {
        'accuracy': test_acc,
        'y_pred': y_pred,
        'y_true': y_true,
        'model': lstm_model
    }

def test_classifier(X_train, X_val, X_test, y_train, y_val, y_test, args, device):
    print("Testing Classifier with pre-trained encoder...")
    
    # Convert data to PyTorch tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Convert one-hot encoded targets to class indices
    if y_train.dim() > 1 and y_train.shape[1] > 1:
        y_train_cls = torch.argmax(y_train, dim=1)
        y_val_cls = torch.argmax(y_val, dim=1)
        y_test_cls = torch.argmax(y_test, dim=1)
    else:
        y_train_cls = y_train.long()
        y_val_cls = y_val.long()
        y_test_cls = y_test.long()
    
    # Build encoder
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    latent_dim = 32
    num_classes = y_train.shape[1] if y_train.dim() > 1 else 1
    
    # Create models
    encoder = build_encoder(input_shape, latent_dim).to(device)
    decoder = build_decoder(latent_dim, input_shape).to(device)
    classifier = build_classifier(latent_dim, num_classes).to(device)
    
    # Define optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    # Define loss functions
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()
    
    # First, train the autoencoder to get good encodings
    print("Pre-training encoder...")
    
    for epoch in range(max(5, args.epochs // 2)):  # Fewer epochs for pre-training
        encoder.train()
        decoder.train()
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        
        ae_loss = 0.0
        
        for i in range(0, len(X_train), args.batch_size):
            # Get batch
            batch_indices = indices[i:i+args.batch_size]
            batch_x = X_train[batch_indices]
            
            # Zero gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            # Forward pass
            encoded = encoder(batch_x)
            decoded = decoder(encoded)
            
            # Calculate loss
            loss = reconstruction_criterion(decoded, batch_x)
            
            # Backward pass and optimize
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            ae_loss += loss.item() * len(batch_x)
        
        ae_loss /= len(X_train)
        
        # Validation
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            encoded_val = encoder(X_val)
            decoded_val = decoder(encoded_val)
            val_loss = reconstruction_criterion(decoded_val, X_val).item()
        
        print(f"Autoencoder Epoch {epoch+1}/{max(5, args.epochs // 2)}, "
              f"Loss: {ae_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Now train the classifier on the encoded data
    print("Training classifier on encoded data...")
    
    # Fix the encoder weights
    for param in encoder.parameters():
        param.requires_grad = False
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(args.epochs):
        classifier.train()
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        
        train_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(0, len(X_train), args.batch_size):
            # Get batch
            batch_indices = indices[i:i+args.batch_size]
            batch_x = X_train[batch_indices]
            batch_y = y_train_cls[batch_indices]
            
            # Zero gradients
            classifier_optimizer.zero_grad()
            
            # Forward pass (encode and classify)
            with torch.no_grad():
                encoded = encoder(batch_x)
            outputs = classifier(encoded)
            
            # Calculate loss
            loss = classification_criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            classifier_optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * len(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(X_train)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        classifier.eval()
        with torch.no_grad():
            encoded_val = encoder(X_val)
            outputs_val = classifier(encoded_val)
            val_loss = classification_criterion(outputs_val, y_val_cls).item()
            _, predicted_val = torch.max(outputs_val.data, 1)
            val_acc = (predicted_val == y_val_cls).sum().item() / len(y_val_cls)
            
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        
        print(f"Classifier Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Evaluate on test set
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        encoded_test = encoder(X_test)
        outputs_test = classifier(encoded_test)
        test_loss = classification_criterion(outputs_test, y_test_cls).item()
        _, predicted_test = torch.max(outputs_test.data, 1)
        test_acc = (predicted_test == y_test_cls).sum().item() / len(y_test_cls)
        
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Get predictions and convert to numpy for evaluation
    y_pred = predicted_test.cpu().numpy()
    y_true = y_test_cls.cpu().numpy()
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Classifier Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Classifier Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/classifier_performance.png")
    
    return {
        'accuracy': test_acc,
        'y_pred': y_pred,
        'y_true': y_true,
        'encoder': encoder,
        'classifier': classifier
    }

def test_all_components(X_train, X_val, X_test, y_train, y_val, y_test, args, device):
    print("Testing all components and full model...")
    
    # Convert data to PyTorch tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Create the full hybrid model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1] if y_train.dim() > 1 else 1
    
    model = SmartGridSecurityModel(
        input_shape=input_shape,
        latent_dim=32,
        num_classes=num_classes,
        device=device
    )
    
    # Train each component as in the full workflow
    print("Training AutoEncoder...")
    ae_history = model.train_autoencoder(X_train, X_val, epochs=args.epochs, batch_size=args.batch_size)
    
    print("Training GAN...")
    d_losses, g_losses = model.train_gan(X_train, epochs=args.epochs, batch_size=args.batch_size)
    
    print("Training Classifier...")
    cls_history = model.train_classifier(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size)
    
    print("Training Full Model...")
    full_history = model.train_full_model(X_train, y_train, X_val, y_val, epochs=max(5, args.epochs // 2), batch_size=args.batch_size)
    
    # Evaluate
    results = model.evaluate(X_test, y_test)
    print("\nFull Model Evaluation:")
    print(f"Classification Report:\n{results['classification_report']}")
    print(f"Anomaly Detection AUC: {results['anomaly_detection_auc']:.4f}")
    
    # Train standalone CNN and LSTM models for comparison
    print("\nTraining standalone CNN for comparison...")
    cnn_results = test_cnn(X_train, X_val, X_test, y_train, y_val, y_test, args, device)
    
    print("\nTraining standalone LSTM for comparison...")
    lstm_results = test_lstm(X_train, X_val, X_test, y_train, y_val, y_test, args, device)
    
    # Convert to numpy for comparison
    if isinstance(y_test, torch.Tensor):
        if y_test.dim() > 1 and y_test.shape[1] > 1:
            y_true = torch.argmax(y_test, dim=1).cpu().numpy()
        else:
            y_true = y_test.cpu().numpy()
    else:
        if y_test.ndim > 1 and y_test.shape[1] > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test
    
    # Extract predictions from full model
    y_pred_proba = model.classify_attacks(X_test)
    hybrid_pred = np.argmax(y_pred_proba, axis=1)

    '''# Create a comparison bar chart
    plt.figure(figsize=(10, 6))
    models = ['Hybrid', 'CNN', 'LSTM']
    accuracies = [hybrid_accuracy, cnn_accuracy, lstm_accuracy]

    plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
    plt.ylim(0, 1.0)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')

    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/model_comparison.png")'''
    
    # Extract predictions from full model
    y_pred_proba = model.classify_attacks(X_test)
    hybrid_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate accuracies
    hybrid_accuracy = accuracy_score(y_true, hybrid_pred)
    cnn_accuracy = accuracy_score(cnn_results['y_true'], cnn_results['y_pred'])
    lstm_accuracy = accuracy_score(lstm_results['y_true'], lstm_results['y_pred'])
    
    # Debug output to verify values
    print(f"DEBUG - Accuracies: Hybrid={hybrid_accuracy:.4f}, CNN={cnn_accuracy:.4f}, LSTM={lstm_accuracy:.4f}")

    # Create a comparison bar chart with dynamic scaling
    create_improved_model_comparison(args.results_dir, hybrid_accuracy, cnn_accuracy, lstm_accuracy)
    
    '''# Create a comparison bar chart
    plt.figure(figsize=(10, 6))
    models = ['Hybrid', 'CNN', 'LSTM']
    accuracies = [hybrid_accuracy, cnn_accuracy, lstm_accuracy]
    
    plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
    plt.ylim(0, 1.0)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/model_comparison.png")'''
    
    # Calculate anomaly detection metrics
    anomalies, mse, threshold = model.detect_anomalies(X_test)
    
    # Create binary labels for anomaly detection (assuming class 0 is normal)
    binary_true = np.zeros(len(y_true))
    binary_true[y_true != 0] = 1
    
    # ROC curve for anomaly detection
    plt.figure(figsize=(10, 8))
    
    # ROC for hybrid model using reconstruction error
    fpr, tpr, _ = roc_curve(binary_true, mse)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Hybrid (AUC = {roc_auc:.4f})')
    
    # Generate soft predictions from CNN and LSTM for ROC
    cnn_probs = torch.softmax(cnn_results['model'](X_test), dim=1).detach().cpu().numpy()[:, 0]
    lstm_probs = torch.softmax(lstm_results['model'](X_test), dim=1).detach().cpu().numpy()[:, 0]
    
    # Invert probabilities since class 0 is normal
    fpr, tpr, _ = roc_curve(binary_true, 1 - cnn_probs)
    cnn_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'CNN (AUC = {cnn_auc:.4f})')
    
    fpr, tpr, _ = roc_curve(binary_true, 1 - lstm_probs)
    lstm_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'LSTM (AUC = {lstm_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve Comparison - Anomaly Detection')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"{args.results_dir}/roc_comparison.png")
    
    # Component analysis summary
    with open(f"{args.results_dir}/component_analysis.txt", 'w') as f:
        f.write("SMART GRID SECURITY MODEL - COMPONENT ANALYSIS\n")
        f.write("===========================================\n\n")
        
        f.write("1. CLASSIFICATION ACCURACY\n")
        f.write("-----------------------\n")
        f.write(f"Hybrid Model: {hybrid_accuracy:.4f}\n")
        f.write(f"CNN-only:     {cnn_accuracy:.4f}\n")
        f.write(f"LSTM-only:    {lstm_accuracy:.4f}\n\n")
        
        f.write("2. ANOMALY DETECTION (AUC)\n")
        f.write("------------------------\n")
        f.write(f"Hybrid Model: {roc_auc:.4f}\n")
        f.write(f"CNN-only:     {cnn_auc:.4f}\n")
        f.write(f"LSTM-only:    {lstm_auc:.4f}\n\n")
        
        f.write("3. CLASSIFICATION REPORT - HYBRID MODEL\n")
        f.write("------------------------------------\n")
        f.write(results['classification_report'])
        f.write("\n\n")
        
        f.write("4. CLASSIFICATION REPORT - CNN-ONLY\n")
        f.write("---------------------------------\n")
        f.write(classification_report(cnn_results['y_true'], cnn_results['y_pred']))
        f.write("\n\n")
        
        f.write("5. CLASSIFICATION REPORT - LSTM-ONLY\n")
        f.write("----------------------------------\n")
        f.write(classification_report(lstm_results['y_true'], lstm_results['y_pred']))
    
    return {
        'hybrid_accuracy': hybrid_accuracy,
        'cnn_accuracy': cnn_accuracy,
        'lstm_accuracy': lstm_accuracy,
        'hybrid_auc': roc_auc,
        'cnn_auc': cnn_auc,
        'lstm_auc': lstm_auc,
        'model': model,
        'anomalies': anomalies,
        'threshold': threshold
    }


def main():
    # Parse arguments
    args = parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set device (use MPS if available on Apple Silicon)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using synthetic data instead...")
        from utils.data_loader import generate_synthetic_data
        df = generate_synthetic_data(n_samples=5000)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)
    
    # Run the appropriate test
    if args.component == 'autoencoder':
        results = test_autoencoder(X_train, X_val, X_test, args, device)
        print(f"AutoEncoder test results:")
        print(f"  - Reconstruction loss: {results['loss']:.6f}")
        print(f"  - Anomaly threshold: {results['threshold']:.6f}")
    
    elif args.component == 'gan':
        results = test_gan(X_train, X_val, X_test, args, device)
        print(f"GAN test results:")
        print(f"  - Discriminator loss: {results['d_loss']:.6f}")
        print(f"  - Generator loss: {results['g_loss']:.6f}")
    
    elif args.component == 'cnn':
        results = test_cnn(X_train, X_val, X_test, y_train, y_val, y_test, args, device)
        print(f"CNN test results:")
        print(f"  - Accuracy: {results['accuracy']:.4f}")
        print(f"  - Classification report:\n{classification_report(results['y_true'], results['y_pred'])}")
    
    elif args.component == 'lstm':
        results = test_lstm(X_train, X_val, X_test, y_train, y_val, y_test, args, device)
        print(f"LSTM test results:")
        print(f"  - Accuracy: {results['accuracy']:.4f}")
        print(f"  - Classification report:\n{classification_report(results['y_true'], results['y_pred'])}")
    
    elif args.component == 'classifier':
        results = test_classifier(X_train, X_val, X_test, y_train, y_val, y_test, args, device)
        print(f"Classifier test results:")
        print(f"  - Accuracy: {results['accuracy']:.4f}")
        print(f"  - Classification report:\n{classification_report(results['y_true'], results['y_pred'])}")
    
    elif args.component == 'all':
        results = test_all_components(X_train, X_val, X_test, y_train, y_val, y_test, args, device)
        print(f"Model comparison results:")
        print(f"  - Hybrid accuracy: {results['hybrid_accuracy']:.4f}")
        print(f"  - CNN accuracy: {results['cnn_accuracy']:.4f}")
        print(f"  - LSTM accuracy: {results['lstm_accuracy']:.4f}")
        print(f"  - Hybrid anomaly detection AUC: {results['hybrid_auc']:.4f}")
        print(f"  - CNN anomaly detection AUC: {results['cnn_auc']:.4f}")
        print(f"  - LSTM anomaly detection AUC: {results['lstm_auc']:.4f}")
        print(f"Detailed analysis saved to {args.results_dir}/component_analysis.txt")

if __name__ == "__main__":
    main()
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import argparse
import os

from model.encoder import build_encoder
from model.decoder import build_decoder
from model.discriminator import build_discriminator
from model.classifier import build_classifier
from model.autoencoder_gan import SmartGridSecurityModel
from utils.data_loader import load_data, preprocess_data

def parse_args():
    parser = argparse.ArgumentParser(description='Test individual components of the hybrid model')
    parser.add_argument('--data_path', type=str, default='./dataset/data/smart_grid_data.csv', help='Path to data file')
    parser.add_argument('--component', type=str, required=True, 
                        choices=['autoencoder', 'gan', 'cnn', 'lstm', 'classifier', 'all'],
                        help='Component to test')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--results_dir', type=str, default='component_tests', help='Directory to save results')
    return parser.parse_args()

def test_autoencoder(X_train, X_val, X_test, args, device):
    print("Testing AutoEncoder component...")
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    latent_dim = 32
    
    # Build encoder and decoder
    encoder = build_encoder(input_shape, latent_dim).to(device)
    decoder = build_decoder(latent_dim, input_shape).to(device)
    
    # Define optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Convert data to PyTorch tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        
    # Train autoencoder
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        
        # Shuffle training data
        indices = torch.randperm(len(X_train))
        
        # Training loop
        train_loss = 0.0
        for i in range(0, len(X_train), args.batch_size):
            # Get batch
            batch_indices = indices[i:i+args.batch_size]
            batch_x = X_train[batch_indices]
            
            # Zero gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            # Forward pass
            encoded = encoder(batch_x)
            decoded = decoder(encoded)
            
            # Calculate loss
            loss = criterion(decoded, batch_x)
            
            # Backward pass and optimize
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            # Update loss
            train_loss += loss.item() * len(batch_x)
        
        train_loss /= len(X_train)
        train_losses.append(train_loss)
        
        # Validation
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            encoded_val = encoder(X_val)
            decoded_val = decoder(encoded_val)
            val_loss = criterion(decoded_val, X_val).item()
            val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Evaluate on test set
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        encoded_test = encoder(X_test)
        decoded_test = decoder(encoded_test)
        test_loss = criterion(decoded_test, X_test).item()
        print(f"Test reconstruction loss (MSE): {test_loss:.6f}")
    
    # Calculate reconstruction error
    with torch.no_grad():
        mse = torch.mean(torch.square(X_test - decoded_test), dim=(1, 2)).cpu().numpy()
        threshold = np.mean(mse) + 3 * np.std(mse)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot training history
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('AutoEncoder Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    
    # Plot sample reconstructions
    n_samples = 5
    samples = X_test[:n_samples].cpu().numpy()
    reconstructed = decoded_test[:n_samples].cpu().numpy()
    
    for i in range(n_samples):
        # Original
        plt.subplot(n_samples, 2, 2*i + 1)
        plt.plot(samples[i, :, 0])
        if i == 0:
            plt.title('Original')
        
        # Reconstructed
        plt.subplot(n_samples, 2, 2*i + 2)
        plt.plot(reconstructed[i, :, 0])
        if i == 0:
            plt.title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/autoencoder_results.png")
    
    return {
        'loss': test_loss,
        'mse': mse,
        'threshold': threshold,
        'encoder': encoder,
        'decoder': decoder
    }

def test_gan(X_train, X_val, X_test, args, device):
    print("Testing GAN component...")
    
    # Create models
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    latent_dim = 32
    
    # Build encoder and discriminator
    encoder = build_encoder(input_shape, latent_dim).to(device)
    discriminator = build_discriminator(latent_dim).to(device)
    
    # Define optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    
    # Define loss function
    criterion = nn.BCELoss()
    
    # Convert data to PyTorch tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Training GAN
    d_losses = []
    g_losses = []
    
    for epoch in range(args.epochs):
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), args.batch_size):
            # Get batch
            batch_indices = indices[i:i+args.batch_size]
            batch_x = X_train[batch_indices]
            batch_size_actual = len(batch_x)
            
            # Create labels
            real_labels = torch.ones(batch_size_actual, 1, device=device)
            fake_labels = torch.zeros(batch_size_actual, 1, device=device)
            
            # -----------------
            # Train Discriminator
            # -----------------
            discriminator_optimizer.zero_grad()
            
            # Real samples
            with torch.no_grad():
                real_encoded = encoder(batch_x)
            real_outputs = discriminator(real_encoded.detach())
            d_loss_real = criterion(real_outputs, real_labels)
            d_loss_real.backward()
            
            # Fake samples (noise)
            noise = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_outputs = discriminator(noise)
            d_loss_fake = criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()
            
            discriminator_optimizer.step()
            
            d_loss = (d_loss_real.item() + d_loss_fake.item()) / 2
            d_loss_epoch += d_loss * batch_size_actual
            
            # -----------------
            # Train Generator (Encoder)
            # -----------------
            encoder_optimizer.zero_grad()
            
            # Get latent representations
            encoded = encoder(batch_x)
            validity = discriminator(encoded)
            
            # Train encoder to fool discriminator
            g_loss = criterion(validity, real_labels)
            g_loss.backward()
            encoder_optimizer.step()
            
            g_loss_epoch += g_loss.item() * batch_size_actual
        
        # Calculate average losses
        d_loss_epoch /= len(X_train)
        g_loss_epoch /= len(X_train)
        
        d_losses.append(d_loss_epoch)
        g_losses.append(g_loss_epoch)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} [D loss: {d_loss_epoch:.4f}] [G loss: {g_loss_epoch:.4f}]")
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.title('GAN Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{args.results_dir}/gan_loss.png")
    
    # Generate latent representations for validation data
    encoder.eval()
    with torch.no_grad():
        encoded_val = encoder(X_val)
        encoded_val = encoded_val.cpu().numpy()
    
    # Visualize latent space
    plt.figure(figsize=(10, 8))
    plt.scatter(encoded_val[:, 0], encoded_val[:, 1], alpha=0.5)
    plt.title('GAN Latent Space (First 2 Dimensions)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(f"{args.results_dir}/gan_latent_space.png")
    
    return {
        'd_loss': d_losses[-1],
        'g_loss': g_losses[-1],
        'encoder': encoder,
        'discriminator': discriminator
    }

def test_cnn(X_train, X_val, X_test, y_train, y_val, y_test, args, device):
    print("Testing CNN component...")
    
    # Convert data to PyTorch tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Convert one-hot encoded targets to class indices
    if y_train.dim() > 1 and y_train.shape[1] > 1:
        y_train_cls = torch.argmax(y_train, dim=1)
        y_val_cls = torch.argmax(y_val, dim=1)
        y_test_cls = torch.argmax(y_test, dim=1)
    else:
        y_train_cls = y_train.long()
        y_val_cls = y_val.long()
        y_test_cls = y_test.long()
    
    # Define CNN model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    num_classes = y_train.shape[1] if y_train.dim() > 1 else 1
    
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            
            # CNN layers
            self.cnn_layers = nn.Sequential(
                nn.Conv1d(input_shape[1], 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()
            )
            
            # Calculate output size after CNN layers
            cnn_output_size = 64 * (input_shape[0] // 4)
            
            # Dense layers
            self.fc_layers = nn.Sequential(
                nn.Linear(cnn_output_size, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            # Transpose for CNN (batch_size, features, time_steps)
            x = x.permute(0, 2, 1)
            x = self.cnn_layers(x)
            x = self.fc_layers(x)
            return x
    
    # Create model and move to device
    cnn_model = CNNModel().to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(args.epochs):
        cnn_model.train()
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        
        train_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(0, len(X_train), args.batch_size):
            # Get batch
            batch_indices = indices[i:i+args.batch_size]
            batch_x = X_train[batch_indices]
            batch_y = y_train_cls[batch_indices]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = cnn_model(batch_x)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * len(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(X_train)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        cnn_model.eval()
        