# Smart Grid Security Model (PyTorch)

A hybrid deep learning approach that simultaneously learns feature representations, detects anomalies, and classifies different types of attacks in smart grids using a combination of AutoEncoder-GAN, CNN, and LSTM. Optimized for Apple Silicon and other modern architectures.

## Architecture

This model combines three powerful deep learning techniques:
- **AutoEncoder-GAN**: For unsupervised feature learning and anomaly detection
- **CNN**: For spatial feature extraction from multivariate grid data
- **LSTM**: For capturing temporal dependencies in time-series grid measurements

## Features

- Anomaly detection based on reconstruction error
- Classification of multiple attack types
- Unsupervised feature learning
- Apple Silicon optimization (Metal Performance Shaders support)
- Synthetic data generation for testing and demonstration
- Comprehensive visualization tools
- Component testing framework

## Project Structure

```
smart-grid-security/
├── main.py                 # Main script to run the model
├── component_test.py       # Script to test individual components
├── run_tests.sh            # Shell script to run all component tests
├── dataset/                # Dataset will go here
│   ├── preprocess_dataset.py # Preprocess dataset first
├── model/                  # Model components
│   ├── __init__.py
│   ├── encoder.py          # Encoder module (CNN-LSTM)
│   ├── decoder.py          # Decoder module
│   ├── discriminator.py    # Discriminator module
│   ├── classifier.py       # Classifier module
│   └── autoencoder_gan.py  # Complete model integration
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_loader.py      # Data loading and preprocessing
│   └── visualization.py    # Visualization utilities
├── tests/                  # Unit tests
│   ├── __init__.py
├── saved_models/           # Directory for saved models
├── results/                # Directory for results and visualizations
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/smart-grid-security.git
cd smart-grid-security
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download database:
```
Intrusion detection evaluation dataset (CIC-IDS2017) is too large to upload to GitHub
Download at the end of this link: https://www.unb.ca/cic/datasets/ids-2017.html
```

### Training

To train the model:

```bash
python main.py --mode train --data_path path/to/your/data.csv --epochs 100
```

### Testing

To evaluate the model:

```bash
python main.py --mode test --data_path path/to/your/data.csv --model_path saved_models/smart_grid_model
```

### Anomaly Detection

To perform anomaly detection on new data:

```bash
python main.py --mode anomaly_detection --data_path path/to/new/data.csv --model_path saved_models/smart_grid_model
```

## Testing Individual Components

You can test individual components of the model to understand their contributions:

```bash
# Test just the autoencoder
python component_test.py --component autoencoder --data_path your_data.csv

# Test just the GAN
python component_test.py --component gan --data_path your_data.csv

# Test just the CNN
python component_test.py --component cnn --data_path your_data.csv

# Test just the LSTM
python component_test.py --component lstm --data_path your_data.csv

# Test just the classifier
python component_test.py --component classifier --data_path your_data.csv

# Test and compare all components
python component_test.py --component all --data_path your_data.csv
```

Or use the provided shell script to run all tests sequentially:

```bash
chmod +x run_tests.sh
./run_tests.sh
```

## Apple Silicon Optimization

This implementation is optimized for Apple Silicon (M1/M2/M3) chips and will automatically use Metal Performance Shaders (MPS) for acceleration when available. If MPS is not available, it will fall back to CPU.

The model checks for MPS availability on startup:
```python
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

## Data Format

The expected input data should be a CSV file with the following structure:
- Multiple columns representing measurements at different time steps
- A final column indicating the class label (0 for normal, 1+ for different attack types)

If no data is provided, the system will generate synthetic data for demonstration purposes.

## Results

After training and evaluation, the results will be stored in the `results/` directory, including:
- Training performance metrics
- Latent space visualization
- Reconstruction error distribution
- Confusion matrix for attack classification

## Demo

To run a quick demonstration using synthetic data:

```bash
python main.py --mode train
```

## Preprocessing Data

If you have your own dataset, you may need to preprocess it first. A sample preprocessing script is provided in `utils/data_loader.py`.

The model expects data in the format `(samples, time_steps, features)` where:
- `samples` is the number of data points
- `time_steps` is the number of sequential time steps per sample
- `features` is the number of features per time step

## License

This project is licensed under the MIT License - see the LICENSE file for details.
