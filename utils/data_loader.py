import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(data_path, test_size=0.2, val_size=0.25):
    """
    Load and preprocess smart grid data
    
    Args:
        data_path: Path to the CSV file containing the data
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using synthetic data for demonstration...")
        df = generate_synthetic_data()
    
    return preprocess_data(df, test_size, val_size)

def preprocess_data(df, test_size=0.2, val_size=0.25):
    """
    Preprocess the data for the model
    
    Args:
        df: DataFrame containing the data
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Assuming the last column is the label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Reshape X for time series (assuming time steps are in consecutive columns)
    # This depends on your specific data structure
    n_samples = X.shape[0]
    n_features = 10  # Adjust based on your data
    time_steps = X.shape[1] // n_features
    
    X = X.reshape(n_samples, time_steps, n_features)
    
    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))
    
    # Split into train and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=42, stratify=y_train_val
    )
    
    # Normalize data
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    # Reshape back
    X_train = X_train_flat.reshape(X_train.shape)
    X_val = X_val_flat.reshape(X_val.shape)
    X_test = X_test_flat.reshape(X_test.shape)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def generate_synthetic_data(n_samples=1000, time_steps=24, n_features=10, n_classes=5):
    """
    Generate synthetic smart grid data for testing
    
    Args:
        n_samples: Number of samples to generate
        time_steps: Number of time steps per sample
        n_features: Number of features per time step
        n_classes: Number of attack classes (including normal)
        
    Returns:
        DataFrame with synthetic data
    """
    # Generate normal data
    normal_data = np.random.normal(0, 1, (n_samples // n_classes, time_steps, n_features))
    
    # Generate attack data with anomalies
    attack_data = []
    for i in range(1, n_classes):
        # Each attack type has a different pattern
        attack = np.random.normal(0, 1, (n_samples // n_classes, time_steps, n_features))
        
        # Add anomalies based on attack type
        if i == 1:  # DDoS
            attack[:, :, 0] += np.random.normal(3, 1, (n_samples // n_classes, time_steps))
        elif i == 2:  # Data Injection
            attack[:, 10:15, 1:3] += np.random.normal(2, 1, (n_samples // n_classes, 5, 2))
        elif i == 3:  # Command Injection
            attack[:, 5:10, 5:7] += np.random.normal(2.5, 1, (n_samples // n_classes, 5, 2))
        elif i == 4:  # Scanning
            attack[:, :, 8:] += np.random.normal(1.5, 1, (n_samples // n_classes, time_steps, 2))
            
        attack_data.append(attack)
    
    # Combine data
    X = np.vstack([normal_data] + attack_data)
    
    # Create labels
    y = np.hstack([
        np.zeros(n_samples // n_classes),
        np.concatenate([np.full(n_samples // n_classes, i) for i in range(1, n_classes)])
    ])
    
    # Shuffle data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Flatten X for DataFrame
    X_flat = X.reshape(X.shape[0], -1)
    
    # Create DataFrame
    columns = [f'feature_{i}_{j}' for i in range(time_steps) for j in range(n_features)]
    df = pd.DataFrame(X_flat, columns=columns)
    df['label'] = y
    
    return df
