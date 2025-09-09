import numpy as np
import torch
import argparse
import os
from model.encoder import build_encoder
from model.decoder import build_decoder
from model.discriminator import build_discriminator
from model.classifier import build_classifier
from model.autoencoder_gan import SmartGridSecurityModel
from utils.data_loader import load_data, preprocess_data
from utils.visualization import plot_results, visualize_latent_space

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Smart Grid Security Model (PyTorch)')
    parser.add_argument('--data_path', type=str, default='dataset/data/smart_grid_data.csv', 
                        help='Path to the dataset')
    parser.add_argument('--model_path', type=str, default='saved_models/smart_grid_model', 
                        help='Path to save/load models')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs for training')
    parser.add_argument('--latent_dim', type=int, default=32, 
                        help='Dimension of the latent space')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'anomaly_detection'],
                        help='Mode of operation')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set device (use MPS if available on Apple Silicon)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.data_path)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Define model parameters
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    num_classes = y_train.shape[1]
    
    # Create model
    model = SmartGridSecurityModel(
        input_shape=input_shape,
        latent_dim=args.latent_dim,
        num_classes=num_classes,
        device=device
    )
    
    if args.mode == 'train':
        # Training workflow
        print("Training AutoEncoder...")
        ae_history = model.train_autoencoder(X_train, X_val, epochs=args.epochs, batch_size=args.batch_size)
        
        print("Training GAN...")
        d_losses, g_losses = model.train_gan(X_train, epochs=args.epochs, batch_size=args.batch_size)
        
        print("Training Classifier...")
        cls_history = model.train_classifier(X_train, y_train, X_val, y_val, 
                                            epochs=args.epochs, batch_size=args.batch_size)
        
        print("Training Full Model...")
        full_history = model.train_full_model(X_train, y_train, X_val, y_val, 
                                             epochs=args.epochs, batch_size=args.batch_size)
        
        # Save model
        model.save_model(args.model_path)
        
        # Visualize results
        plot_results(ae_history, d_losses, g_losses, cls_history, full_history)
        
    elif args.mode == 'test':
        # Load model
        model.load_model(args.model_path)
        
        # Evaluate
        results = model.evaluate(X_test, y_test)
        print("Classification Report:")
        print(results['classification_report'])
        print("\nAnomaly Detection AUC:", results['anomaly_detection_auc'])
        
        # Visualize latent space
        visualize_latent_space(model, X_test, y_test)
        
    elif args.mode == 'anomaly_detection':
        # Load model
        model.load_model(args.model_path)
        
        # Detect anomalies
        anomalies, mse, threshold = model.detect_anomalies(X_test)
        print(f"Detected {np.sum(anomalies)} anomalies out of {len(X_test)} samples")
        print(f"Threshold: {threshold}")
        
        # Classify detected anomalies
        if np.sum(anomalies) > 0:
            anomalous_data = X_test[anomalies]
            y_pred = model.classify_attacks(anomalous_data)
            print("Attack type distribution:")
            attack_types = ['Normal', 'DDoS', 'Data Injection', 'Command Injection', 'Scanning']
            for i in range(len(attack_types)):
                count = np.sum(np.argmax(y_pred, axis=1) == i)
                print(f"{attack_types[i]}: {count}")

if __name__ == "__main__":
    main()
