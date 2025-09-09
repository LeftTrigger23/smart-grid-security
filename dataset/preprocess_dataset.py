import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
import glob
from tqdm import tqdm

def preprocess_cicids2017_for_smart_grid(input_folder, output_file, time_steps=24, n_features=10):
    """
    Preprocess CICIDS2017 dataset for the smart grid security model
    
    Args:
        input_folder: Path to the folder containing CICIDS2017 CSV files
        output_file: Path to save the preprocessed file
        time_steps: Number of time steps for each sample
        n_features: Number of features to select
    """
    print("Step 1: Loading and combining data files...")
    # Find all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return
    
    # Load and combine all CSV files
    dataframes = []
    for file in tqdm(csv_files):
        try:
            # Read CSV with appropriate settings
            df = pd.read_csv(file, encoding='latin1', low_memory=False)
            # Standardize column names
            df.columns = df.columns.str.strip().str.replace(' ', '_')
            dataframes.append(df)
            print(f"  Loaded {file}, shape: {df.shape}")
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    
    if not dataframes:
        print("No data loaded. Please check the input files.")
        return
    
    # Combine all dataframes
    print("Step 2: Combining dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"  Combined data shape: {combined_df.shape}")
    
    # Identify label column (might be 'Label' or 'label')
    label_col = None
    for col in ['Label', 'label']:
        if col in combined_df.columns:
            label_col = col
            break
    
    if not label_col:
        print("Error: Could not find label column ('Label' or 'label')")
        return
    
    print(f"Step 3: Processing attack labels from column '{label_col}'...")
    # Map attack types to numeric values
    attack_mapping = {
        'BENIGN': 0,
        'DDoS': 1,
        'PortScan': 2,
        'Bot': 3,
        'Infiltration': 3,
        'Web Attack': 4,
        'Web Attack - Brute Force': 4,
        'Web Attack - XSS': 4,
        'Web Attack - Sql Injection': 4,
        'FTP-Patator': 4,
        'SSH-Patator': 4,
        'DoS slowloris': 1,
        'DoS Slowhttptest': 1,
        'DoS Hulk': 1,
        'DoS GoldenEye': 1,
        'Heartbleed': 4
    }
    
    # Handle label column - use mapping for known attack types
    combined_df['new_label'] = combined_df[label_col].apply(
        lambda x: attack_mapping.get(
            x.strip() if isinstance(x, str) else x,
            attack_mapping.get(
                x.split(' ')[0].strip() if isinstance(x, str) and ' ' in x else x, 
                4  # Default for unknown attacks
            )
        )
    )
    
    # Check label distribution
    print("  Attack distribution:")
    for label, count in combined_df['new_label'].value_counts().items():
        print(f"    Label {label}: {count} samples ({count/len(combined_df)*100:.2f}%)")
    
    print("Step 4: Cleaning and preprocessing data...")
    # Drop original label column and any non-numeric columns
    combined_df = combined_df.drop(label_col, axis=1)
    
    # Identify numeric columns
    numeric_cols = combined_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'new_label']
    
    print(f"  Identified {len(numeric_cols)} numeric feature columns")
    
    # Handle missing and infinite values
    combined_df = combined_df[numeric_cols + ['new_label']]
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values with column means
    for col in numeric_cols:
        if combined_df[col].isna().any():
            col_mean = combined_df[col].mean()
            combined_df[col] = combined_df[col].fillna(col_mean)
    
    print("Step 5: Selecting top features based on correlation with the label...")
    # Calculate correlation with label for feature selection
    correlations = []
    for col in numeric_cols:
        try:
            corr = abs(combined_df[col].corr(combined_df['new_label']))
            if not np.isnan(corr):
                correlations.append((col, corr))
        except Exception as e:
            print(f"  Error calculating correlation for {col}: {e}")
    
    # Sort by correlation and select top features
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = [col for col, corr in correlations[:n_features]]
    
    print(f"  Selected top {len(top_features)} features:")
    for i, (col, corr) in enumerate(correlations[:n_features]):
        print(f"    {i+1}. {col} (correlation: {corr:.4f})")
    
    # Select only the top features and the label
    selected_df = combined_df[top_features + ['new_label']]
    
    print("Step 6: Creating time series windows...")
    # Scale features
    scaler = MinMaxScaler()
    selected_df[top_features] = scaler.fit_transform(selected_df[top_features])
    
    # Create time windows
    n_samples = len(selected_df) // time_steps
    print(f"  Creating {n_samples} samples with {time_steps} time steps each")
    
    # Truncate data to fit exactly into n_samples
    truncated_df = selected_df.iloc[:n_samples * time_steps]
    
    # Extract features and labels
    X = truncated_df[top_features].values.reshape(n_samples, time_steps, n_features)
    y = truncated_df['new_label'].values.reshape(n_samples, time_steps)
    # Use the most frequent label in each window
    y = np.array([np.bincount(window).argmax() for window in y])
    
    print("Step 7: Preparing final dataset...")
    # Flatten the 3D array to 2D for saving to CSV
    X_flat = X.reshape(n_samples, time_steps * n_features)
    
    # Create column names for the flattened features
    col_names = [f'feature_{i}_{j}' for i in range(time_steps) for j in range(n_features)]
    
    # Create final dataframe
    final_df = pd.DataFrame(X_flat, columns=col_names)
    final_df['label'] = y
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"Step 8: Saved preprocessed dataset to {output_file}")
    print(f"  Final dataset shape: {final_df.shape}")
    print(f"  Final attack distribution:")
    for label, count in final_df['label'].value_counts().items():
        print(f"    Label {label}: {count} samples ({count/len(final_df)*100:.2f}%)")
    
    print("Preprocessing complete!")

# Example usage
if __name__ == "__main__":
    preprocess_cicids2017_for_smart_grid(
        input_folder="./MachineLearningCVE",
        output_file="data/smart_grid_data.csv",
        time_steps=24,
        n_features=10
    )