import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Extracted data from the image
expected_returns = np.array([0.095, 0.101, 0.091, 0.109, 0.14, 0.157])  # Expected returns
sustainability = np.array([0.0001,0.0002, 0.0003, 0.104, 0.1005, 0.1006])
# Correlation matrix extracted from the image
correlation_matrix = np.array([
    [1.00, 0.82, 0.93, 0.20, 0.23, 0.22],
    [0.82, 1.00, 0.92, 0.37, 0.43, 0.45],
    [0.93, 0.92, 1.00, 0.29, 0.34, 0.34],
    [0.20, 0.37, 0.29, 1.00, 0.88, 0.80],
    [0.23, 0.43, 0.34, 0.88, 1.00, 0.93],
    [0.22, 0.45, 0.34, 0.80, 0.93, 1.00]
])

# Constructing the variance-covariance matrix (sigma_matrix)
sigma_matrix = np.array([
    [13.5, 0, 0, 0, 0, 0],
    [8.2, 5.6, 0, 0, 0, 0],
    [9.1, 2.7, 2.4, 0, 0, 0],
    [3.7, 6.3, 0.3, 16.5, 0, 0],
    [3.6, 6.8, 0.3, 11.7, 7.3, 0],
    [3.6, 7.7, 0.1, 10.4, 6.8, 5.9],
])/100
# Multiplying sigma matrix with its transpose
sigma_matrix_transpose = sigma_matrix.T
result_matrix = np.sqrt(np.matmul(sigma_matrix, sigma_matrix_transpose))

# Extracting the diagonal of the result matrix to create a numpy array
volatilities = np.diag(result_matrix)
risk_free_rate = 0.05


# Calculate portfolio return
def portfolio_return(weights):
    return np.dot(weights, expected_returns)

# Calculate portfolio variance
def portfolio_variance(weights):
    return np.dot(weights.T, np.dot(np.dot(sigma_matrix,sigma_matrix_transpose), weights))

# Calculate portfolio standard deviation (risk)
def portfolio_risk(weights):
    return np.sqrt(portfolio_variance(weights))

def portfolio_sustainability(weights):
    return np.dot(weights, sustainability)

# Objective function: minimize the negative Sharpe ratio (equivalently, minimize variance for a given return)
def objective_function(weights, target_return):
    return portfolio_variance(weights)# Here we might later incorporate a condition for a target return

# Constraints for the optimization: weights sum to 1, and achieve target return
def constraint_sum_to_one(weights):
    return np.sum(weights) - 1

def constraint_target_return(weights, target_return):
    return portfolio_return(weights) - target_return

# Range of target returns for which we will optimize the portfolio to build the efficient frontier
target_returns = np.linspace(0, 0.4, 100)
risk_values = []
return_values = []

# Perform optimization for each target return
for target_return in target_returns:
    constraints = (
        {'type': 'eq', 'fun': constraint_sum_to_one},
        {'type': 'eq', 'fun': lambda w: constraint_target_return(w, target_return)},
    )
    result = minimize(
        fun=objective_function,
        x0=np.array([1/len(expected_returns)]*len(expected_returns)),  # Initial guess: equal weights
        args=(target_return,),
        method='SLSQP',
        constraints=constraints,
        bounds=[(-10000, 10000) for _ in range(len(expected_returns))],  # Weights between 0 and 1
    )
    if result.success:
        optimal_weights = result.x
        risk_values.append(portfolio_risk(optimal_weights))
        return_values.append(target_return)


# Plotting the efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(risk_values, return_values, 'r--', label='Efficient Frontier')
for i in range(len(expected_returns)):
    plt.scatter(volatilities[i], expected_returns[i], label=f'Asset {i+1}')

plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.xlim(0, 0.4)
plt.ylim(0, 0.4)
plt.title('Efficient Frontier')
plt.legend()
plt.show()
