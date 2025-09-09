import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import torch

def plot_results(ae_history, d_losses, g_losses, cls_history, full_history):
    """
    Plot training results
    
    Args:
        ae_history: Dictionary with autoencoder training history
        d_losses: List of discriminator losses
        g_losses: List of generator losses
        cls_history: Dictionary with classifier training history
        full_history: Dictionary with full model training history
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot autoencoder loss
    axes[0, 0].plot(ae_history['loss'], label='train')
    axes[0, 0].plot(ae_history['val_loss'], label='validation')
    axes[0, 0].set_title('Autoencoder Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].legend()
    
    # Plot GAN losses
    axes[0, 1].plot(d_losses, label='discriminator')
    axes[0, 1].plot(g_losses, label='generator')
    axes[0, 1].set_title('GAN Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Plot classifier accuracy
    axes[1, 0].plot(cls_history['accuracy'], label='train')
    axes[1, 0].plot(cls_history['val_accuracy'], label='validation')
    axes[1, 0].set_title('Classifier Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    
    # Plot full model loss
    axes[1, 1].plot(full_history['loss'], label='total loss')
    axes[1, 1].plot(full_history['decoder_output_loss'], label='reconstruction')
    axes[1, 1].plot(full_history['classifier_output_loss'], label='classification')
    axes[1, 1].set_title('Full Model Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('results/training_results.png')
    plt.close()

def visualize_latent_space(model, X, y, perplexity=30):
    """
    Visualize the latent space using t-SNE
    
    Args:
        model: Trained model
        X: Input data (PyTorch tensor)
        y: Labels (PyTorch tensor)
        perplexity: t-SNE perplexity parameter
    """
    # Ensure model is in evaluation mode
    model.encoder.eval()
    
    # Move data to the same device as model
    if isinstance(X, torch.Tensor):
        X = X.to(model.device)
    else:
        X = torch.tensor(X, dtype=torch.float32).to(model.device)
    
    # Generate encoded data
    with torch.no_grad():
        encoded_data = model.encoder(X).cpu().numpy()
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    encoded_tsne = tsne.fit_transform(encoded_data)
    
    # Convert one-hot encoded labels to integers
    if isinstance(y, torch.Tensor):
        if y.dim() > 1 and y.shape[1] > 1:
            y_labels = torch.argmax(y, dim=1).cpu().numpy()
        else:
            y_labels = y.cpu().numpy()
    else:
        if y.ndim > 1 and y.shape[1] > 1:
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y
    
    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(encoded_tsne[:, 0], encoded_tsne[:, 1], c=y_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Attack Type')
    plt.title('t-SNE Visualization of Latent Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('results/latent_space_visualization.png')
    plt.close()

def plot_reconstruction_error(mse, threshold, y_true):
    """
    Plot reconstruction error distribution
    
    Args:
        mse: Reconstruction error values
        threshold: Anomaly detection threshold
        y_true: True labels
    """
    # Convert one-hot encoded labels to integers if needed
    if isinstance(y_true, torch.Tensor):
        if y_true.dim() > 1 and y_true.shape[1] > 1:
            y_labels = torch.argmax(y_true, dim=1).cpu().numpy()
        else:
            y_labels = y_true.cpu().numpy()
    else:
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_labels = np.argmax(y_true, axis=1)
        else:
            y_labels = y_true
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot reconstruction error distribution
    sns.histplot(mse, bins=50, kde=True, alpha=0.6)
    
    # Plot threshold line
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    
    # Plot error distribution by class
    for i in range(np.max(y_labels) + 1):
        sns.kdeplot(mse[y_labels == i], label=f'Class {i}')
    
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/reconstruction_error.png')
    plt.close()

def plot_confusion_matrix(conf_matrix, class_names=None):
    """
    Plot confusion matrix
    
    Args:
        conf_matrix: Confusion matrix from sklearn
        class_names: List of class names
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(conf_matrix.shape[0])]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()

def visualize_gan_latent_space_tsne(encoded_data, labels=None, perplexity=30, n_components=2, results_dir="results"):
    """
    Visualize GAN latent space using t-SNE dimensionality reduction
    
    Args:
        encoded_data: Encoded latent representations
        labels: Optional labels for color-coding (can be None)
        perplexity: t-SNE perplexity parameter
        n_components: Number of dimensions for t-SNE output
        results_dir: Directory to save results
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Apply t-SNE for dimensionality reduction
    print("Applying t-SNE to reduce dimensionality...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, 
                learning_rate='auto', init='pca')
    encoded_tsne = tsne.fit_transform(encoded_data)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot differently based on whether labels are provided
    if labels is not None:
        # Convert one-hot encoded labels to class indices if needed
        if labels.ndim > 1 and labels.shape[1] > 1:
            y_labels = np.argmax(labels, axis=1)
        else:
            y_labels = labels.flatten()
        
        # Define attack type names for the legend
        attack_types = ['Normal', 'DDoS', 'Data Injection', 'Command Injection', 'Scanning']
        
        # Create scatter plot with color-coding by class
        scatter = plt.scatter(encoded_tsne[:, 0], encoded_tsne[:, 1], 
                            c=y_labels, cmap='viridis', alpha=0.7, s=50)
        
        # Add legend with attack type names
        legend1 = plt.legend(scatter.legend_elements()[0], 
                          [attack_types[i] for i in range(len(np.unique(y_labels)))],
                          title="Attack Types", loc="upper right")
        plt.gca().add_artist(legend1)
    else:
        # Simple scatter plot without color-coding
        plt.scatter(encoded_tsne[:, 0], encoded_tsne[:, 1], alpha=0.7, s=50)
    
    # Add title and labels
    plt.title('t-SNE Visualization of GAN Latent Space (32D → 2D)', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add colorbar if color-coded
    if labels is not None:
        cbar = plt.colorbar(scatter)
        cbar.set_label('Attack Type Class')
    
    # Improve aesthetics
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{results_dir}/gan_latent_space_tsne.png", dpi=300, bbox_inches='tight')
    print(f"t-SNE visualization saved as '{results_dir}/gan_latent_space_tsne.png'")
    
    return encoded_tsne