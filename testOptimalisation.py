# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:28:28 2024

@author: joche
"""
import cvxpy as cp
import numpy as np

def optimize_portfolio(returns, cov_matrix, risk_aversion):
    """
    Solves the portfolio optimization problem.
    
    :param returns: An array of expected returns for each asset.
    :param cov_matrix: The covariance matrix for the assets.
    :param risk_aversion: A scalar that determines the trade-off between risk and return.
    :return: Optimal portfolio weights.
    """

    # Number of assets
    n_assets = len(returns)
    
    # Portfolio weights variable
    weights = cp.Variable(n_assets)
    
    # Expected portfolio return
    portfolio_return = returns.T @ weights
    
    # Portfolio variance
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    
    # Objective function: Maximize return and penalize variance
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
    
    # Constraints
    constraints = [
        cp.sum(weights) == 1,  # Sum of weights is 1
        weights >= 0            # No short selling (weights are non-negative)
    ]
    
    # Problem
    problem = cp.Problem(objective, constraints)
    
    # Solve the problem
    problem.solve()
    
    # Return the optimal weights
    return weights.value

# Example data
expected_returns = np.array([0.05, 0.06, 0.07])  # Expected returns for 3 assets
covariance_matrix = np.array([[0.1, 0.01, 0.02],
                              [0.01, 0.1, 0.03],
                              [0.02, 0.03, 0.1]])  # Covariance matrix for the assets
risk_aversion_factor = 0.5  # Risk aversion

# Optimize portfolio
optimal_weights = optimize_portfolio(expected_returns, covariance_matrix, risk_aversion_factor)

print("Optimal portfolio weights:", optimal_weights)
