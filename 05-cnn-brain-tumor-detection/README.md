\# 05 — CNN — Brain Tumor Detection (MRI Classification)

A from-scratch NumPy implementation of a CNN, compared against an equivalent TensorFlow/Keras model, classifying brain MRI scans as tumor / no tumor.



\## Dataset

Brain MRI Images for Brain Tumor Detection (Kaggle: `navoneel/brain-mri-images-for-brain-tumor-detection`), grayscale images resized to 32×32. Binary classification, \~61/39 tumor/no-tumor split, stratified train/test split to preserve that ratio.



\## Architecture

\- \*\*Input\*\*: 32×32 grayscale MRI image, standardized.

\- \*\*Scratch model\*\*: Conv (4 filters, 3×3, valid, ReLU) → MaxPool (2×2) → Flatten → Dense(1, sigmoid), implemented explicitly in NumPy with full forward/backward passes, trained with per-sample online SGD on binary cross-entropy loss.

\- \*\*Keras model\*\*: `Conv2D(4, 3, relu)` → `MaxPooling2D(2)` → `Flatten` → `Dense(1, sigmoid)`, trained with the Adam optimizer.

\- \*\*Output\*\*: Binary classification (tumor / no tumor).

\- Learning rate: 0.001, Epochs: 30



\## Results

| Metric | Scratch (NumPy) | TensorFlow/Keras |

|--------|------------------|-------------------|

| Test Accuracy | 0.7843 | 0.8235 |

| True Positives | 27 | 26 |

| True Negatives | 13 | 16 |

| False Positives | 7 | 4 |

| False Negatives | 4 | 5 |

| Epochs | 30 | 30 |

| Filters | 4 | 4 |

| Kernel size | 3×3 | 3×3 |

| Pool size | 2×2 | 2×2 |



!\[Scratch CNN loss curve](assets/loss\_curve\_scratch.png)

\*Training loss (scratch NumPy model) over 30 epochs.\*



!\[Keras CNN loss curve](assets/loss\_curve\_keras.png)

\*Training loss (Keras model) over 30 epochs — faster, smoother convergence due to Adam's adaptive learning rate.\*



\## How to run

1\. Open `cnn.ipynb` in Google Colab.

2\. Run all cells — the dataset downloads automatically via `kagglehub` on first run. Note: unlike the other projects' CSV auto-downloads, this may prompt for Kaggle sign-in/API access the first time you run it in a fresh Colab session.

