#!/bin/bash
# Script to run all component tests for the smart grid security model

# Create results directories
mkdir -p results/autoencoder
mkdir -p results/gan
mkdir -p results/cnn
mkdir -p results/lstm
mkdir -p results/classifier
mkdir -p results/full_model

# Set common parameters
DATA_PATH="data/smart_grid_data.csv"
EPOCHS=20
BATCH_SIZE=32

echo "===== SMART GRID SECURITY MODEL - COMPONENT TESTING ====="
echo ""

# Test AutoEncoder component
echo "1/6: Testing AutoEncoder component..."
python component_test.py --component autoencoder --data_path $DATA_PATH --epochs $EPOCHS --batch_size $BATCH_SIZE --results_dir results/autoencoder

# Test GAN component
echo ""
echo "2/6: Testing GAN component..."
python component_test.py --component gan --data_path $DATA_PATH --epochs $EPOCHS --batch_size $BATCH_SIZE --results_dir results/gan

# Test CNN component
echo ""
echo "3/6: Testing CNN component..."
python component_test.py --component cnn --data_path $DATA_PATH --epochs $EPOCHS --batch_size $BATCH_SIZE --results_dir results/cnn

# Test LSTM component
echo ""
echo "4/6: Testing LSTM component..."
python component_test.py --component lstm --data_path $DATA_PATH --epochs $EPOCHS --batch_size $BATCH_SIZE --results_dir results/lstm

# Test Classifier component
echo ""
echo "5/6: Testing Classifier component..."
python component_test.py --component classifier --data_path $DATA_PATH --epochs $EPOCHS --batch_size $BATCH_SIZE --results_dir results/classifier

# Test Full Model with all components
echo ""
echo "6/6: Testing Full Model with all components..."
python component_test.py --component all --data_path $DATA_PATH --epochs $EPOCHS --batch_size $BATCH_SIZE --results_dir results/full_model

echo ""
echo "===== TESTING COMPLETE ====="
echo "Results saved in the 'results' directory"