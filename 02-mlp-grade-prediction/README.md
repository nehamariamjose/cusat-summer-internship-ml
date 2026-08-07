# 02 -- Multi-Layer Perceptron (MLP)

Regression model predicting final grade (G3) for students, using an MLP trained 
from scratch (NumPy) and compared against a TensorFlow/Keras implementation.

## Dataset
Student Performance dataset (`student-mat.csv`), using features `studytime`, 
`failures`, `absences`, `G1`, `G2` to predict `G3`.

## Architecture
- Input → Dense(10, ReLU) → Dense(5, ReLU) → Dense(1, Linear)
- Learning rate: 0.01, Epochs: 2000

## Results

| Metric | Scratch (NumPy) | TensorFlow/Keras |
|--------|------------------|-------------------|
| MSE    | 4.8123           | 5.1125            |
| MAE    | 1.5383           | 1.1727           |

![Loss Curve — Scratch](assets/loss_curve_scratch.png)
*Training loss (scratch NumPy model) over 2000 epochs — converges smoothly via plain gradient descent.*

![Loss Curve — Keras](assets/loss_curve_keras.png)
*Training loss (Keras model) over 2000 epochs — faster convergence due to Adam's adaptive learning rate.*

![Predictions — Scratch](assets/predictions_scatter_scratch.png)
*Predicted vs. actual G3 grades (scratch model) on the test set. Points closer to the red dashed line indicate better predictions.*

![Predictions — Keras](assets/predictions_scatter_keras.png)
*Predicted vs. actual G3 grades (Keras model) on the test set.*

## How to run
1. Open `mlp.ipynb` in Google Colab.
2. Run all cells — the dataset downloads automatically from this repo.