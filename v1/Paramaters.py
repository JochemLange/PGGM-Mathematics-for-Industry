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
