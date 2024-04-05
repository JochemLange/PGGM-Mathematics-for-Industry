# Extracted data from the image
expected_returns = np.array([0.095, 0.101, 0.091, 0.109, 0.14, 0.157])  # Expected returns

esg_ratings = np.array([
    85, # Gov. bonds
    60, # BAA bonds
    92, # AAA Bonds
    75, # growth stocks
    60, # Int. Stocks
    70, # Value stocks
])/100

sigma_matrix = np.array([
    [0.135, 0, 0, 0, 0, 0],  # Government Bonds
    [0.082, 0.056, 0, 0, 0, 0],  # Corporate Bonds, Baa
    [0.091, 0.027, 0.024, 0, 0, 0],  # Corporate Bonds, Aaa
    [0.037, 0.063, 0.003, 0.165, 0, 0],  # Growth Stocks
    [0.036, 0.068, 0.003, 0.117, 0.073, 0],  # International Stocks
    [0.036, 0.077, 0.001, 0.104, 0.068, 0.059],  # Value Stocks
])
