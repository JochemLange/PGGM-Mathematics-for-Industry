import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Manually inputting the data.
# Note: For simplification, dictionaries and lists are retained for readability and structure where numpy does not offer a significant advantage.

# Λ (lambda) coefficients for the sources of risk
lambda_coeffs = np.array([0.331, 0.419, -0.029, 0.126, 0.477, 0.305])

# Σ (covariance) matrix.
sigma_matrix = np.array([
    [13.5, 0, 0, 0, 0, 0],
    [8.2, 5.6, 0, 0, 0, 0],
    [9.1, 2.7, 2.4, 0, 0, 0],
    [3.7, 6.3, 0.3, 16.5, 0, 0],
    [3.6, 6.8, 0.3, 11.7, 7.3, 0],
    [3.6, 7.7, 0.1, 10.4, 6.8, 5.9],
])

# Expected returns for each asset
expected_returns = np.array([9.5, 10.1, 9.1, 10.9, 14.0, 15.7]) / 100  # Convert percentages to decimals

# Correlations matrix (transformed from percentage to decimal)
correlations_matrix = np.array([
    [1.00, 0.82, 0.93, 0.20, 0.23, 0.22],
    [0.82, 1.00, 0.92, 0.37, 0.43, 0.45],
    [0.93, 0.92, 1.00, 0.29, 0.34, 0.34],
    [0.20, 0.37, 0.29, 1.00, 0.88, 0.80],
    [0.23, 0.43, 0.34, 0.88, 1.00, 0.93],
    [0.22, 0.45, 0.34, 0.80, 0.93, 1.00],
])

# Lambda_0 coefficients (Λ0)
lambda_0 = np.array([0.306, 0.409, -0.020, 0.089, 0.498, 0.310, 0, 0, 0])

# Kappa values (κi)
kappa_values = np.array([0.36, 0.12, 0.052])

# Constants
risk_free_rate = 0.05

# Construct the covariance matrix from volatilities and correlations
volatilities = sigma_matrix.diagonal()**0.5 / 100  # Extract volatilities from the diagonal, assuming it represents variances
covariance_matrix = np.outer(volatilities, volatilities) * correlations_matrix
print(covariance_matrix)