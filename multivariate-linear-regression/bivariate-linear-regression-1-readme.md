# Bivariate Linear Regression (Gradient Descent)

This project implements bivariate (two-feature) linear regression using gradient descent from scratch in Python. It’s designed for educational purposes, allowing users to input their own data, experiment with hyperparameters, and visualize the learning process.

## Features

- **Interactive Data Entry:** Enter up to 100 data points via the command line.
- **Feature Normalization:** Ensures stable and efficient gradient descent.
- **Customizable Training:** Set your own learning rate (`alpha`) and number of iterations.
- **Gradient Descent Implementation:** All calculations are performed using NumPy vectorization.
- **Prediction Interface:** Make predictions on new data points after training.
- **Divergence Detection:** Warns if the gradient diverges (NaNs/Infs).
- **Clear Output:** Displays the learned hypothesis and parameter values.

## How It Works

1. **Data Collection:** Enter pairs of `(x1, x2, y)` values interactively.
2. **Normalization:** Features are normalized for better convergence.
3. **Training:** The model trains using user-defined learning rate and iterations.
4. **Output:** The final regression equation is displayed.
5. **Prediction:** The user can input new `x1, x2` values to predict `y`.

## Usage

```bash
python bivariate-linear-regression-1.py
```

You will be prompted to:
- Enter your data points (`x1`, `x2`, `y`)
- Set the learning rate and number of iterations
- View the trained model
- Make new predictions

## Requirements

- Python 3.x
- NumPy

Install NumPy if needed:
```bash
pip install numpy
```

## Example

```
Run the program? (Yes/No or Y/N): Yes

Enter training data (enter "end" to finish.)
Data point #1, enter x₁: 2
Data point #1, enter x₂: 3
Enter the corresponding value for Y: 15
...

Enter the value of Learning Rate (Alpha): 0.01
Enter the number of iterations you'd like to perform: 1500

Calculated Hypothesis is:
H(x) = 1.23 + 4.56·x₁ + 7.89·x₂
The value of parameters are: 1.23000000, 4.56000000, 7.89000000
Start prediction?: (Yes/No or Y/N) Yes
Enter value of x1 for which to predict value of Y: 5
Enter value of x2 for which to predict value of Y: 2

Calculated value of Y is: 48.17
```
