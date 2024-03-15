# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:25:27 2024

@author: joche
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def optimize_portfolio_for_target_return(returns, cov_matrix, target_return):
    """
    Solves for the minimum variance portfolio for a given target return.
    
    :param returns: An array of expected returns for each asset.
    :param cov_matrix: The covariance matrix for the assets.
    :param target_return: The target return for the portfolio.
    :return: Optimal portfolio weights and the portfolio's variance.
    """
    
    n_assets = len(returns)
    weights = cp.Variable(n_assets)
    portfolio_return = returns.T @ weights
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    
    objective = cp.Minimize(portfolio_variance)
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        portfolio_return == target_return
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return weights.value, portfolio_variance.value


def generate_efficient_frontier(returns, cov_matrix, num_points=100):
    """
    Generates the efficient frontier for a given set of assets.
    
    :param returns: An array of expected returns for each asset.
    :param cov_matrix: The covariance matrix for the assets.
    :param num_points: Number of points on the frontier to calculate.
    :return: Arrays of variances and returns for the efficient frontier.
    """
    
    min_return, max_return = np.min(returns), np.max(returns)
    target_returns = np.linspace(min_return, max_return, num_points)
    variances, frontier_returns = [], []
    
    for target_return in target_returns:
        try:
            _, variance = optimize_portfolio_for_target_return(returns, cov_matrix, target_return)
            variances.append(variance)
            frontier_returns.append(target_return)
        except:
            pass  # Skip points where optimization fails
    
    return np.sqrt(variances), frontier_returns

covariance_matrix = np.array([
    [1.00, 0.82, 0.93, 0.20, 0.23, 0.22],
    [0.82, 1.00, 0.92, 0.37, 0.43, 0.45],
    [0.93, 0.92, 1.00, 0.29, 0.34, 0.34],
    [0.20, 0.37, 0.29, 1.00, 0.88, 0.80],
    [0.23, 0.43, 0.34, 0.88, 1.00, 0.93],
    [0.22, 0.45, 0.34, 0.80, 0.93, 1.00],
])

sigma_matrix = np.array([
    [13.5, 0, 0, 0, 0, 0],
    [8.2, 5.6, 0, 0, 0, 0],
    [9.1, 2.7, 2.4, 0, 0, 0],
    [3.7, 6.3, 0.3, 16.5, 0, 0],
    [3.6, 6.8, 0.3, 11.7, 7.3, 0],
    [3.6, 7.7, 0.1, 10.4, 6.8, 5.9],
])

# Lambda_0 coefficients (Λ0)
expected_returns =  np.array([9.5, 10.1, 9.1, 10.9, 14.0, 15.7]) / 100

variances, returns = generate_efficient_frontier(expected_returns, sigma_matrix)

print(variances)

# Plotting the efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(variances, returns, 'o--')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.show()
