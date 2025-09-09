import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from model.encoder import build_encoder
from model.decoder import build_decoder
from model.discriminator import build_discriminator
from model.classifier import build_classifier

class SmartGridSecurityModel:
    def __init__(
        self,
        input_shape=(24, 10),  # (time_steps, features)
        latent_dim=32,
        num_classes=5,  # Number of attack types + normal
        lstm_units=128,
        cnn_filters=[64, 128],
        dropout_rate=0.2,
        learning_rate=0.001,
        device=None
    ):
        """
        Initialize the complete smart grid security model
        
        Args:
            input_shape: Tuple (time_steps, features)
            latent_dim: Dimension of the latent space
            num_classes: Number of attack types (including normal)
            lstm_units: Number of LSTM units
            cnn_filters: List of filter counts for CNN layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizers
            device: PyTorch device (if None, will use MPS if available, else CPU)
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.cnn_filters = cnn_filters
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Set device (use MPS if available on Apple Silicon)
        if device is None:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Build the component models
        self.encoder = build_encoder(self.input_shape, self.latent_dim).to(self.device)
        self.decoder = build_decoder(self.latent_dim, self.input_shape).to(self.device)
        self.discriminator = build_discriminator(self.latent_dim, self.dropout_rate).to(self.device)
        self.classifier = build_classifier(self.latent_dim, self.num_classes, self.dropout_rate).to(self.device)
        
        # Setup optimizers
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        self.classifier_optimizer = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        
        # Loss functions
        self.reconstruction_criterion = nn.MSELoss()
        self.adversarial_criterion = nn.BCELoss()
        self.classification_criterion = nn.CrossEntropyLoss()
        
    def train_autoencoder(self, X_train, X_val, epochs=100, batch_size=32):
        """
        Train the autoencoder component
        
        Args:
            X_train: Training data
            X_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Dictionary with training history
        """
        # Convert data to PyTorch tensors if needed
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
        
        n_batches = len(X_train) // batch_size
        history = {
            'loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        self.encoder.train()
        self.decoder.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle training data
            indices = torch.randperm(len(X_train))
            
            # Training loop
            for i in range(n_batches):
                # Get batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                batch_indices = indices[start_idx:end_idx]
                batch_x = X_train[batch_indices].to(self.device)
                
                # Zero gradients
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                
                # Forward pass
                encoded = self.encoder(batch_x)
                decoded = self.decoder(encoded)
                
                # Calculate loss
                loss = self.reconstruction_criterion(decoded, batch_x)
                
                # Backward pass and optimize
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                
                # Update epoch loss
                epoch_loss += loss.item()
            
            # Calculate average epoch loss
            epoch_loss /= n_batches
            history['loss'].append(epoch_loss)
            
            # Validation
            with torch.no_grad():
                self.encoder.eval()
                self.decoder.eval()
                
                val_x = X_val.to(self.device)
                val_encoded = self.encoder(val_x)
                val_decoded = self.decoder(val_encoded)
                val_loss = self.reconstruction_criterion(val_decoded, val_x).item()
                
                history['val_loss'].append(val_loss)
                
                self.encoder.train()
                self.decoder.train()
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best models
                torch.save(self.encoder.state_dict(), 'encoder_best.pth')
                torch.save(self.decoder.state_dict(), 'decoder_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Load best models
                    self.encoder.load_state_dict(torch.load('encoder_best.pth'))
                    self.decoder.load_state_dict(torch.load('decoder_best.pth'))
                    break
        
        return history
    
    def train_gan(self, X_train, epochs=100, batch_size=32):
        """
        Train the GAN component
        
        Args:
            X_train: Training data
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Discriminator and generator losses
        """
        # Convert data to PyTorch tensors if needed
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        
        n_batches = len(X_train) // batch_size
        d_losses = []
        g_losses = []
        
        # Labels for real and fake
        real_label = 1.0
        fake_label = 0.0
        
        for epoch in range(epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            
            # Shuffle training data
            indices = torch.randperm(len(X_train))
            
            for i in range(n_batches):
                # Get batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                batch_indices = indices[start_idx:end_idx]
                batch_x = X_train[batch_indices].to(self.device)
                batch_size_actual = len(batch_x)
                
                # -----------------
                # Train Discriminator
                # -----------------
                self.discriminator_optimizer.zero_grad()
                
                # Real samples
                real_encoded = self.encoder(batch_x)
                real_labels = torch.full((batch_size_actual, 1), real_label, 
                                        dtype=torch.float, device=self.device)
                real_outputs = self.discriminator(real_encoded.detach())
                d_loss_real = self.adversarial_criterion(real_outputs, real_labels)
                d_loss_real.backward()
                
                # Fake samples
                noise = torch.randn(batch_size_actual, self.latent_dim, device=self.device)
                fake_labels = torch.full((batch_size_actual, 1), fake_label, 
                                        dtype=torch.float, device=self.device)
                fake_outputs = self.discriminator(noise)
                d_loss_fake = self.adversarial_criterion(fake_outputs, fake_labels)
                d_loss_fake.backward()
                
                # Update discriminator
                d_loss = d_loss_real + d_loss_fake
                self.discriminator_optimizer.step()
                
                # -----------------
                # Train Generator (Encoder)
                # -----------------
                self.encoder_optimizer.zero_grad()
                
                # Generate encoded samples
                encoded = self.encoder(batch_x)
                # Discriminator should classify these as real
                outputs = self.discriminator(encoded)
                # Generator's goal is to make discriminator classify as real
                g_loss = self.adversarial_criterion(outputs, real_labels)
                g_loss.backward()
                
                # Update generator
                self.encoder_optimizer.step()
                
                # Update losses
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
            
            # Calculate average epoch losses
            epoch_d_loss /= n_batches
            epoch_g_loss /= n_batches
            d_losses.append(epoch_d_loss)
            g_losses.append(epoch_g_loss)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} [D loss: {epoch_d_loss:.4f}] [G loss: {epoch_g_loss:.4f}]")
        
        return d_losses, g_losses
    
    def train_classifier(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the classifier component
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Dictionary with training history
        """
        # Convert data to PyTorch tensors if needed
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
        
        # Convert one-hot encoded targets to class indices
        if y_train.dim() > 1 and y_train.shape[1] > 1:
            y_train = torch.argmax(y_train, dim=1)
            y_val = torch.argmax(y_val, dim=1)
            
        y_train = y_train.long()
        y_val = y_val.long()
        
        n_batches = len(X_train) // batch_size
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            self.encoder.eval()  # We don't train the encoder here
            self.classifier.train()
            
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            # Shuffle training data
            indices = torch.randperm(len(X_train))
            
            # Training loop
            for i in range(n_batches):
                # Get batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                batch_indices = indices[start_idx:end_idx]
                batch_x = X_train[batch_indices].to(self.device)
                batch_y = y_train[batch_indices].to(self.device)
                
                # Zero gradients
                self.classifier_optimizer.zero_grad()
                
                # Forward pass
                with torch.no_grad():
                    encoded = self.encoder(batch_x)
                outputs = self.classifier(encoded)
                
                # Calculate loss
                loss = self.classification_criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                self.classifier_optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            # Calculate epoch metrics
            epoch_loss /= n_batches
            accuracy = correct / total
            history['loss'].append(epoch_loss)
            history['accuracy'].append(accuracy)
            
            # Validation
            with torch.no_grad():
                self.classifier.eval()
                
                val_x = X_val.to(self.device)
                val_y = y_val.to(self.device)
                
                val_encoded = self.encoder(val_x)
                val_outputs = self.classifier(val_encoded)
                val_loss = self.classification_criterion(val_outputs, val_y).item()
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_correct = (val_predicted == val_y).sum().item()
                val_accuracy = val_correct / len(val_y)
                
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                self.classifier.train()
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {accuracy:.4f}, " 
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.classifier.state_dict(), 'classifier_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Load best model
                    self.classifier.load_state_dict(torch.load('classifier_best.pth'))
                    break
        
        return history
    
    def train_full_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the full model end-to-end
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Dictionary with training history
        """
        # Convert data to PyTorch tensors if needed
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
        
        # Convert one-hot encoded targets to class indices for classifier
        if y_train.dim() > 1 and y_train.shape[1] > 1:
            y_train_cls = torch.argmax(y_train, dim=1).long()
            y_val_cls = torch.argmax(y_val, dim=1).long()
        else:
            y_train_cls = y_train.long()
            y_val_cls = y_val.long()
        
        n_batches = len(X_train) // batch_size
        history = {
            'loss': [],
            'decoder_output_loss': [],
            'discriminator_output_loss': [],
            'classifier_output_loss': [],
            'classifier_output_accuracy': [],
            'val_loss': [],
            'val_decoder_output_loss': [],
            'val_discriminator_output_loss': [],
            'val_classifier_output_loss': [],
            'val_classifier_output_accuracy': []
        }
        
        # Set loss weights
        loss_weights = {
            'decoder': 1.0,
            'discriminator': 0.5,
            'classifier': 1.0
        }
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Set all components to training mode
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        self.classifier.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_decoder_loss = 0.0
            epoch_discriminator_loss = 0.0
            epoch_classifier_loss = 0.0
            correct = 0
            total = 0
            
            # Shuffle training data
            indices = torch.randperm(len(X_train))
            
            for i in range(n_batches):
                # Get batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                batch_indices = indices[start_idx:end_idx]
                batch_x = X_train[batch_indices].to(self.device)
                batch_y_cls = y_train_cls[batch_indices].to(self.device)
                
                # Real labels for discriminator
                real_labels = torch.ones(len(batch_x), 1, device=self.device)
                
                # Zero gradients
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()
                self.classifier_optimizer.zero_grad()
                
                # Forward pass
                encoded = self.encoder(batch_x)
                decoded = self.decoder(encoded)
                validity = self.discriminator(encoded)
                class_pred = self.classifier(encoded)
                
                # Calculate losses
                decoder_loss = self.reconstruction_criterion(decoded, batch_x)
                discriminator_loss = self.adversarial_criterion(validity, real_labels)
                classifier_loss = self.classification_criterion(class_pred, batch_y_cls)
                
                # Weighted combined loss
                loss = (loss_weights['decoder'] * decoder_loss + 
                        loss_weights['discriminator'] * discriminator_loss + 
                        loss_weights['classifier'] * classifier_loss)
                
                # Backward pass and optimize
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                self.discriminator_optimizer.step()
                self.classifier_optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                epoch_decoder_loss += decoder_loss.item()
                epoch_discriminator_loss += discriminator_loss.item()
                epoch_classifier_loss += classifier_loss.item()
                
                _, predicted = torch.max(class_pred.data, 1)
                total += batch_y_cls.size(0)
                correct += (predicted == batch_y_cls).sum().item()
            
            # Calculate epoch metrics
            epoch_loss /= n_batches
            epoch_decoder_loss /= n_batches
            epoch_discriminator_loss /= n_batches
            epoch_classifier_loss /= n_batches
            accuracy = correct / total
            
            history['loss'].append(epoch_loss)
            history['decoder_output_loss'].append(epoch_decoder_loss)
            history['discriminator_output_loss'].append(epoch_discriminator_loss)
            history['classifier_output_loss'].append(epoch_classifier_loss)
            history['classifier_output_accuracy'].append(accuracy)
            
            # Validation
            with torch.no_grad():
                self.encoder.eval()
                self.decoder.eval()
                self.discriminator.eval()
                self.classifier.eval()
                
                val_x = X_val.to(self.device)
                val_y_cls = y_val_cls.to(self.device)
                real_labels_val = torch.ones(len(val_x), 1, device=self.device)
                
                val_encoded = self.encoder(val_x)
                val_decoded = self.decoder(val_encoded)
                val_validity = self.discriminator(val_encoded)
                val_class_pred = self.classifier(val_encoded)
                
                val_decoder_loss = self.reconstruction_criterion(val_decoded, val_x).item()
                val_discriminator_loss = self.adversarial_criterion(val_validity, real_labels_val).item()
                val_classifier_loss = self.classification_criterion(val_class_pred, val_y_cls).item()
                
                val_loss = (loss_weights['decoder'] * val_decoder_loss + 
                            loss_weights['discriminator'] * val_discriminator_loss + 
                            loss_weights['classifier'] * val_classifier_loss)
                
                _, val_predicted = torch.max(val_class_pred.data, 1)
                val_correct = (val_predicted == val_y_cls).sum().item()
                val_accuracy = val_correct / len(val_y_cls)
                
                history['val_loss'].append(val_loss)
                history['val_decoder_output_loss'].append(val_decoder_loss)
                history['val_discriminator_output_loss'].append(val_discriminator_loss)
                history['val_classifier_output_loss'].append(val_classifier_loss)
                history['val_classifier_output_accuracy'].append(val_accuracy)
                
                self.encoder.train()
                self.decoder.train()
                self.discriminator.train()
                self.classifier.train()
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best models
                torch.save(self.encoder.state_dict(), 'encoder_best.pth')
                torch.save(self.decoder.state_dict(), 'decoder_best.pth')
                torch.save(self.discriminator.state_dict(), 'discriminator_best.pth')
                torch.save(self.classifier.state_dict(), 'classifier_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Load best models
                    self.encoder.load_state_dict(torch.load('encoder_best.pth'))
                    self.decoder.load_state_dict(torch.load('decoder_best.pth'))
                    self.discriminator.load_state_dict(torch.load('discriminator_best.pth'))
                    self.classifier.load_state_dict(torch.load('classifier_best.pth'))
                    break
        
        return history
    
    def detect_anomalies(self, X_test, threshold=None):
        """
        Detect anomalies using reconstruction error
        
        Args:
            X_test: Test data
            threshold: Threshold for anomaly detection (if None, calculated automatically)
            
        Returns:
            Tuple of (anomalies, mse, threshold)
        """
        # Convert data to PyTorch tensor if needed
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        
        # Set models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        
        # Get predictions
        with torch.no_grad():
            X_test = X_test.to(self.device)
            encoded = self.encoder(X_test)
            X_pred = self.decoder(encoded)
            
            # Calculate reconstruction error
            mse = torch.mean(torch.square(X_test - X_pred), dim=(1, 2))
            mse = mse.cpu().numpy()
        
        # Calculate threshold if not provided
        if threshold is None:
            threshold = np.mean(mse) + 3 * np.std(mse)
        
        # Flag anomalies
        anomalies = mse > threshold
        
        return anomalies, mse, threshold
    
    def classify_attacks(self, X_test):
        """
        Classify attack types
        
        Args:
            X_test: Test data
            
        Returns:
            Class probabilities
        """
        # Convert data to PyTorch tensor if needed
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        
        # Set models to evaluation mode
        self.encoder.eval()
        self.classifier.eval()
        
        # Get predictions
        with torch.no_grad():
            X_test = X_test.to(self.device)
            encoded = self.encoder(X_test)
            y_pred = self.classifier(encoded)
            y_pred = F.softmax(y_pred, dim=1)
            y_pred = y_pred.cpu().numpy()
        
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert data to PyTorch tensors if needed
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
        
        # Convert one-hot encoded targets to class indices if needed
        if y_test.dim() > 1 and y_test.shape[1] > 1:
            y_true = torch.argmax(y_test, dim=1).cpu().numpy()
        else:
            y_true = y_test.cpu().numpy()
        
        # Detect anomalies
        anomalies, mse, threshold = self.detect_anomalies(X_test)
        
        # Classify attacks
        y_pred_proba = self.classify_attacks(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        report = classification_report(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Calculate ROC curve for anomaly detection
        normal_idx = np.where(y_true == 0)[0]  # Assuming class 0 is normal
        attack_idx = np.where(y_true != 0)[0]
        
        # Create binary labels for anomaly detection
        y_binary = np.zeros(len(y_true))
        y_binary[attack_idx] = 1
        
        fpr, tpr, _ = roc_curve(y_binary, mse)
        roc_auc = auc(fpr, tpr)
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'anomaly_detection_auc': roc_auc,
            'reconstruction_error': mse,
            'threshold': threshold
        }
    
    def save_model(self, path):
        """
        Save model components
        
        Args:
            path: Path prefix for saving model files
        """
        torch.save(self.encoder.state_dict(), f"{path}_encoder.pth")
        torch.save(self.decoder.state_dict(), f"{path}_decoder.pth")
        torch.save(self.discriminator.state_dict(), f"{path}_discriminator.pth")
        torch.save(self.classifier.state_dict(), f"{path}_classifier.pth")
        
        # Save model configuration
        model_config = {
            'input_shape': self.input_shape,
            'latent_dim': self.latent_dim,
            'num_classes': self.num_classes,
            'lstm_units': self.lstm_units,
            'cnn_filters': self.cnn_filters,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }
        torch.save(model_config, f"{path}_config.pth")
        
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model components
        
        Args:
            path: Path prefix for loading model files
        """
        # Load model configuration
        model_config = torch.load(f"{path}_config.pth")
        self.input_shape = model_config['input_shape']
        self.latent_dim = model_config['latent_dim']
        self.num_classes = model_config['num_classes']
        self.lstm_units = model_config['lstm_units']
        self.cnn_filters = model_config['cnn_filters']
        self.dropout_rate = model_config['dropout_rate']
        self.learning_rate = model_config['learning_rate']
        
        # Rebuild models with the loaded configuration
        self.encoder = build_encoder(self.input_shape, self.latent_dim).to(self.device)
        self.decoder = build_decoder(self.latent_dim, self.input_shape).to(self.device)
        self.discriminator = build_discriminator(self.latent_dim, self.dropout_rate).to(self.device)
        self.classifier = build_classifier(self.latent_dim, self.num_classes, self.dropout_rate).to(self.device)
        
        # Load weights
        self.encoder.load_state_dict(torch.load(f"{path}_encoder.pth", map_location=self.device))
        self.decoder.load_state_dict(torch.load(f"{path}_decoder.pth", map_location=self.device))
        self.discriminator.load_state_dict(torch.load(f"{path}_discriminator.pth", map_location=self.device))
        self.classifier.load_state_dict(torch.load(f"{path}_classifier.pth", map_location=self.device))
        
        # Set models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        self.classifier.eval()
        
        print(f"Model loaded from {path}")