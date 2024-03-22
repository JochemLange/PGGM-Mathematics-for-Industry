import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Extracted data from the image
expected_returns = np.array([0.095, 0.101, 0.091, 0.109, 0.14, 0.157, 0.07, 0.065, 0.08, 0.045, 0.12, 0.09])  # Expected returns

esg_ratings = np.array([
    85, # Gov. bonds
    60, # BAA bonds
    92, # AAA Bonds
    75, # growth stocks
    60, # Int. Stocks
    70, # Value stocks
    70,  # Commercial Real Estate
    75,  # Residential Real Estate
    80,  # REITs
    55,  # Gold
    45,  # Oil
    85   # Agricultural Products
])/100

sigma_matrix = np.array([
    [0.135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Government Bonds
    [0.082, 0.056, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Corporate Bonds, Baa
    [0.091, 0.027, 0.024, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Corporate Bonds, Aaa
    [0.037, 0.063, 0.003, 0.165, 0, 0, 0, 0, 0, 0, 0, 0],  # Growth Stocks
    [0.036, 0.068, 0.003, 0.117, 0.073, 0, 0, 0, 0, 0, 0, 0],  # International Stocks
    [0.036, 0.077, 0.001, 0.104, 0.068, 0.059, 0, 0, 0, 0, 0, 0],  # Value Stocks
    [0.04, 0.04, 0.035, 0.05, 0.05, 0.045, 0.08, 0, 0, 0, 0, 0],  # Commercial Real Estate
    [0.038, 0.038, 0.033, 0.048, 0.048, 0.043, 0.06, 0.075, 0, 0, 0, 0],  # Residential Real Estate
    [0.042, 0.042, 0.037, 0.055, 0.055, 0.05, 0.065, 0.06, 0.085, 0, 0, 0],  # REITs
    [0.02, 0.02, 0.018, 0.03, 0.03, 0.025, 0.04, 0.038, 0.04, 0.06, 0, 0],  # Gold
    [0.05, 0.05, 0.045, 0.07, 0.07, 0.065, 0.08, 0.078, 0.08, 0.05, 0.1, 0],  # Oil
    [0.03, 0.03, 0.025, 0.04, 0.04, 0.035, 0.05, 0.048, 0.05, 0.04, 0.06, 0.07],  # Agricultural Products
])

# variance covariane matrix
variance_covariance_matrix = sigma_matrix @ sigma_matrix.T

print(variance_covariance_matrix)
# Calculate standard deviations (sqrt of variances)
std_devs = np.sqrt(np.diag(variance_covariance_matrix))

# Outer product of standard deviation vector with itself gives a matrix
# where each element (i, j) is the product of std_devs[i] and std_devs[j]
std_dev_matrix = np.outer(std_devs, std_devs)

# Calculate correlation matrix by element-wise division
correlation_matrix = variance_covariance_matrix / std_dev_matrix

# Normalize diagonal elements to 1
np.fill_diagonal(correlation_matrix, 1)

correlation_matrix_rounded = np.around(correlation_matrix, decimals=3)

# Extracting the diagonal of the result matrix to create a numpy array
volatilities = np.sqrt(np.diag(variance_covariance_matrix))


# managar i can only trade in assets where there is a 1 in the row i.
manager_distribution = [[1,1,1,0,0,0,0,0,0,0,0,0], [0,0,0,1,1,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,1,1,0,0,0], [0,0,0,0,0,0,0,0,0,1,1,1]]

# Calculate portfolio return
def portfolio_return(weights, expected_returns):
    return np.dot(weights, expected_returns)

# Calculate portfolio variance
def portfolio_variance(weights, variance_covariance_matrix):
    return np.dot(weights.T, np.dot(variance_covariance_matrix, weights))

# Calculate portfolio standard deviation (risk)
def portfolio_risk(weights, variance_covariance_matrix):
    return np.sqrt(portfolio_variance(weights, variance_covariance_matrix))

def portfolio_sustainability(weights, esg_ratings):
    return np.dot(weights, esg_ratings)

risk_free_rate = 0.05
risk_aversion = 10
sustainability_inclination = 0

def objective_function(weights):
    return -(portfolio_return(weights, expected_returns) - (risk_aversion/2) * portfolio_risk(weights, variance_covariance_matrix) + sustainability_inclination * portfolio_sustainability(weights, esg_ratings))

# Constraint for the weights to sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds for each weight (between 0 and 1)
bounds = tuple((0, 1) for _ in range(len(expected_returns)))

# Optimize for each manager
optimal_weights = []
for manager in manager_distribution:
    # Adjusting bounds based on manager's trading limitations
    manager_bounds = [(0, 1) if manager[i] == 1 else (0, 0.000001) for i in range(len(manager))]
    
    # Initial guess (equally distributed within allowed investments)
    initial_guess = np.array(manager) / sum(manager)
    
    # Optimization
    result = minimize(objective_function, initial_guess, method='SLSQP', bounds=manager_bounds, constraints=constraints)
    
    optimal_weights.append(result.x)

print([[round(weight, 3) for weight in manager_weights] for manager_weights in optimal_weights])
