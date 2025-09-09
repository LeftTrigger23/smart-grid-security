import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from model.autoencoder_gan import SmartGridSecurityModel
from utils.data_loader import generate_synthetic_data, preprocess_data
from utils.visualization import plot_reconstruction_error, plot_confusion_matrix, visualize_latent_space

def run_demonstration():
    """
    Run a complete demonstration of the model on synthetic data
    """
    print("Smart Grid Security Model Demonstration")
    print("---------------------------------------")
    
    # 1. Generate synthetic data
    print("\nGenerating synthetic smart grid data...")
    df = generate_synthetic_data(n_samples=5000, time_steps=24, n_features=10, n_classes=5)
    print(f"Generated dataset with {len(df)} samples")
    
    # 2. Preprocess data
    print("\nPreprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)
    
    # 3. Initialize model
    print("\nInitializing model...")
    model = SmartGridSecurityModel(
        input_shape=(24, 10),
        latent_dim=32,
        num_classes=5,
        dropout_rate=0.3
    )
    
    # 4. Training stages
    print("\nTraining autoencoder (Stage 1)...")
    model.train_autoencoder(X_train, X_val, epochs=10, batch_size=32)
    
    print("\nTraining GAN (Stage 2)...")
    d_losses, g_losses = model.train_gan(X_train, epochs=10, batch_size=32)
    
    print("\nTraining classifier (Stage 3)...")
    model.train_classifier(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    
    print("\nFine-tuning full model (Stage 4)...")
    model.train_full_model(X_train, y_train, X_val, y_val, epochs=5, batch_size=32)
    
    # 5. Evaluation
    print("\nEvaluating model performance...")
    results = model.evaluate(X_test, y_test)
    
    print("\nClassification Report:")
    print(results['classification_report'])
    
    print(f"\nAnomaly Detection ROC-AUC: {results['anomaly_detection_auc']:.4f}")
    
    # 6. Anomaly detection example
    print("\nRunning anomaly detection on test set...")
    anomalies, mse, threshold = model.detect_anomalies(X_test)
    print(f"Detected {np.sum(anomalies)} anomalies out of {len(X_test)} samples")
    print(f"Anomaly threshold: {threshold:.4f}")
    
    # 7. Attack classification example
    print("\nClassifying attack types...")
    y_pred_proba = model.classify_attacks(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    attack_types = ['Normal', 'DDoS Attack', 'Data Injection', 'Command Injection', 'Scanning']
    attack_counts = {}
    for i, attack in enumerate(attack_types):
        count = np.sum(y_pred == i)
        attack_counts[attack] = count
        print(f"{attack}: {count} instances")
    
    # 8. Visualize latent space
    print("\nVisualizing latent space...")
    visualize_latent_space(model, X_test, y_test)
    print("Saved latent space visualization to 'results/latent_space_visualization.png'")
    
    # 9. Plot reconstruction error
    print("\nPlotting reconstruction error distribution...")
    plot_reconstruction_error(mse, threshold, y_test)
    print("Saved reconstruction error plot to 'results/reconstruction_error.png'")
    
    # 10. Plot confusion matrix
    print("\nGenerating confusion matrix...")
    conf_matrix = results['confusion_matrix']
    plot_confusion_matrix(conf_matrix, class_names=attack_types)
    print("Saved confusion matrix to 'results/confusion_matrix.png'")
    
    print("\nDemonstration complete!")

if __name__ == "__main__":
    run_demonstration()
