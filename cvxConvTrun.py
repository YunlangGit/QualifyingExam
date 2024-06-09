import cvxpy as cp
import numpy as np
import time

def generate_A_matrices(m, n, k):
    sigma = 1 / np.sqrt(k)
    return np.random.normal(0, sigma, (k,m*n))

# generate the ground truth X_star
def generate_X(m, n, rank):
    U = np.random.randn(m, rank)
    V = np.random.randn(n, rank)
    return U @ V.T

# calculate y
def calculate_y(A, X_star):
    return A @ X_star.flatten()

# parameters
m, n, r, k = 100, 50, 10, 4000

# synthesize data
A = generate_A_matrices(m, n, k)
X_star = generate_X(m, n, r)
y = A @ X_star.flatten()

# define the decision variable
X = cp.Variable((m, n))

# define the objective function (nuclear norm)
objective = cp.Minimize(cp.normNuc(X))

# Define the constraints
constraints = [A @ cp.vec(X.T) == y]

# Form and solve the problem
prob = cp.Problem(objective, constraints)

start = time.time()

# Solve the problem
prob.solve(verbose=True)

# Get the optimized value of X and truncate
X_optimized = X.value
U, S, Vt = np.linalg.svd(X_optimized, full_matrices=False)
Sr = np.zeros_like(S)
Sr[0:r] = S[0:r]
Sr_diag = np.diag(Sr)
X_trun = U @ Sr_diag @ Vt
end = time.time()

print("X Rank:", np.linalg.matrix_rank(X_trun))
print("Error:", np.linalg.norm(X_star-X_optimized))
print("Error after Truncation", np.linalg.norm(X_star-X_trun))
print("Time elapsed:", end - start)
print("Value of ||A(X)-y||", np.linalg.norm(A @ X_trun.flatten() - y))