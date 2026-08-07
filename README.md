# cusat-summer-internship-ml
This repository contains five deep learning architectures implemented from scratch in NumPy, mirrored in TensorFlow/Keras, and documented with accompanying math notes. Built during my ML/DL internship at CUSAT in June 2026.

## Projects

| # | Project | Description |
|---|---------|-------------|
| 1 | [SVD Image Compression](./01-svd-image-compression) | Power iteration + deflation SVD, image reconstruction |
| 2 | [MLP Grade Prediction](./02-mlp-grade-prediction) | Multi-layer perceptron regression on student performance data |
| 3 | [RNN Temperature Forecasting](./03-rnn-temp-forecast) | Vanilla RNN forecasting temperature from a sliding window, trained with BPTT |
| 4 | [LSTM Stock Prediction](./04-lstm-stock-prediction) | LSTM regression predicting next-day AAPL closing price from a 5-day window |
| 5 | [CNN Brain Tumor Detection](./05-cnn-brain-tumor-detection) | CNN classifying brain MRI scans as tumor / no tumor |

## Setup
```bash
pip install -r requirements.txt
```