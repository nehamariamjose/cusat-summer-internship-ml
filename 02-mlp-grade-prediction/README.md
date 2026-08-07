\# 02 — Multi-Layer Perceptron (MLP)



Regression model predicting final grade (G3) for students, using an MLP trained 

from scratch (NumPy) and compared against a TensorFlow/Keras implementation.



\## Dataset

Student Performance dataset (`student-mat.csv`), using features `studytime`, 

`failures`, `absences`, `G1`, `G2` to predict `G3`.



\## Architecture

\- Input → Dense(10, ReLU) → Dense(5, ReLU) → Dense(1, Linear)

\- Learning rate: 0.01, Epochs: 2000



\## Results



| Metric | Scratch (NumPy) | TensorFlow/Keras |

|--------|------------------|-------------------|

| MSE    | 4.8123           | 3.0456            |

| MAE    | 1.5383           | 1.0809            |



!\[Loss Curve — Scratch](assets/loss\_curve\_scratch.png)

!\[Loss Curve — Keras](assets/loss\_curve\_keras.png)

!\[Predictions — Scratch](assets/predictions\_scatter\_scratch.png)

!\[Predictions — Keras](assets/predictions\_scatter\_keras.png)



\## How to run

1\. Open `mlp.ipynb` in Google Colab.

2\. Upload `student-mat.csv` to the Colab session (or update `DATA\_PATH` in the config cell).

3\. Run all cells.

