# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:24:49 2024

@author: jochem
"""

"""
Script with usefull functions!
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, expected_returns, correlation_matrix, sustainability, risk_free_rate, risk_aversion=1.0):
        " Initilize the portofolio optimiser with all required inputs "
        # The 4 important variables: R, sigma, sustainablitity
        self.expected_returns   = expected_returns
        self.correlation_matrix = correlation_matrix
        self.sustainability     = sustainability
        
        self.risk_free_rate     = risk_free_rate
        self.risk_aversion      = risk_aversion
        
        # Calculate the volatility matrix from the given sigma matrix
        self.volatilities = np.sqrt(np.diag(self.correlation_matrix))

    def portfolio_return(self, weights):
        "Return the expected portolio return for specific weights."
        return np.dot(weights, self.expected_returns)

    def portfolio_risk(self, weights):
        "Returns the risk (sigma, standard deviation) of a given portolio with some weights."
        return np.dot(weights.T, np.dot(self.correlation_matrix, weights))


    def portfolio_sustainability(self, weights):
        "Returns the sustainablity of a portifolio."
        return np.dot(weights, self.sustainability)

    def objective_function(self, weights):
        # Adjust this function as needed. Here, minimizing variance as a placeholder.
        return   self.portfolio_risk(weights) 


    def optimize_portfolio(self, target_return=None, initial_guess=None, bounds=(0, 1), constraints=()):
        """
        A function that returns the optimal weights.

        Parameters
        ----------
        target_return : float, optional
            DESCRIPTION. A target The default is None.
        initial_guess : TYPE, optional
            DESCRIPTION. The default is None.
        bounds : TYPE, optional
            DESCRIPTION. The default is (-1, 1).
        constraints : TYPE, optional
            DESCRIPTION. The default is ().

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if initial_guess is None:
            # Iitial guess is spread over all assets.
            initial_guess = np.array([1/len(self.expected_returns)]*len(self.expected_returns))
        
        # Define the constraints including the sum of weights to 1
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        # If a target_return is given we append it as a constrant.
        if target_return is not None:
            cons.append({'type': 'eq', 'fun': lambda w: self.portfolio_return(w) - target_return})

        # Combine user-defined constraints
        cons.extend(constraints)
        
        result = minimize(
            fun=self.objective_function,
            x0=initial_guess,
            method='SLSQP',
            constraints=cons,
            bounds=[bounds for _ in range(len(self.expected_returns))],
        )
        if not result.success:
            return None
            # raise ValueError("Optimization did not converge: ", result.message)
        return result.x

    # Example method to plot the efficient frontier
    def plot_efficient_frontier(self, target_returns):
        risk_values = []
        return_values = []
        bestOptimalsWeights = []
        
        # Big for loop through all target returns.
        tempInitialGuessForFurtherOptimisation = None
        for target_return in target_returns:
            optimal_weights = self.optimize_portfolio(target_return=target_return,initial_guess=tempInitialGuessForFurtherOptimisation)
            # Check if minimisation succeced
            if optimal_weights is not None:
                risk_values.append(np.sqrt(self.portfolio_risk(optimal_weights)))
                return_values.append(target_return)
                
                # To plot the best weights.
                bestOptimalsWeights.append(optimal_weights)
                
                tempInitialGuessForFurtherOptimisation = optimal_weights
        
        
        # You can add your plotting code here
        # This will plot the efficient frontier based on the calculated risk_values and return_values
        plt.figure(figsize=(10, 6))
        plt.plot(risk_values, return_values, 'r--', label='Efficient Frontier')
        for i in range(len(self.expected_returns)):
            plt.scatter(self.volatilities[i], self.expected_returns[i], label=f'Asset {i+1}')

        plt.xlabel('Risk (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.xlim(0, 0.4)
        plt.ylim(0, 0.4)
        plt.title('Efficient Frontier')
        plt.legend()
        plt.show()
        
        
        bestOptimalsWeights = np.array(bestOptimalsWeights)
        
        print(np.shape(bestOptimalsWeights))
        
        plt.figure(figsize=(10, 6))
        for i, weight in enumerate(bestOptimalsWeights.T):
            plt.plot(return_values, weight, label='Assist' + str(i))
        

        plt.xlabel('Return')
        plt.ylabel('How much stock')
        plt.xlim(0, 0.4)
        plt.ylim(0, 1)
        plt.title('Optimal weigth a location')
        plt.legend()
        plt.show()


# Example usage:

# Call optimizer.plot_efficient_frontier(target_returns) to plot the efficient frontier

        
