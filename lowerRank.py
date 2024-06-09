import numpy as np
import matplotlib.pyplot as plt
import time

# generate operator A
def generate_A_matrices(n, k):
    sigma = 1 / np.sqrt(k)
    return np.random.normal(0, sigma, (k,n*n))

# generate the ground truth X_star
def generate_X(n, rank):
    U = np.random.randn(n, rank)
    return U @ U.T

# calculate y
def calculate_y(A, X_star):
    return A @ X_star.flatten()

# gradient calculation
def gradient(U, A, y):
    k = len(A)
    #  initializes a zero matrix of the same shape as U to accumulate the gradient values.
    grad_U = np.zeros_like(U)
    UUT = U @ U.T
    for i in range(k):
        # loop through all A_i matrices
        Ai = A[i]
        residual = Ai @ UUT.flatten() - y[i]
        grad_U += 4 * residual * Ai.reshape(n,n) @ U
    return grad_U / k

def gradient_descent(U, A, y, X_star, max_iter=1000, tol=1e-6):
    alpha = 10
    eta = 0.1
    errors = []
    for iter in range(max_iter):
        f = np.linalg.norm(A @ (U @ U.T).flatten() - y) ** 2
        grad_U = gradient(U, A, y)
        normU = np.linalg.norm(U, 2)
        Utemp = U - alpha * grad_U
        ftemp = np.linalg.norm(A @ (Utemp @ Utemp.T).flatten() - y) ** 2
        while ftemp > f - eta * alpha * np.linalg.norm(grad_U, 'fro') ** 2:
            alpha = 0.5 * alpha
            Utemp = U - alpha * grad_U
            ftemp = np.linalg.norm(A @ (Utemp @ Utemp.T).flatten() - y) ** 2
        U = Utemp
        error = np.linalg.norm(A @ (U @ U.T).flatten() - y) ** 2
        errors.append(error)
        if error < tol or np.linalg.norm(grad_U) < tol:
            break
    return U, errors, iter

# parameters
n, r, k = 50, 10, 2000

print('-----------Start-----------')

# synthesize data
A = generate_A_matrices(n, k)

# random_size = np.random.randint(r) + 1
random_size = 1

print('Rank of X_star is', random_size)
X_star = generate_X(n, random_size)
y = calculate_y(A, X_star)

# initialize U and V
size = 1
U1 = np.random.randn(n, size)

start = time.time()
while size <= r:
    print("current size:", U1.shape[1])
    # # initial error from X_star
    # initial_error = np.linalg.norm(U0 @ V0.T - X_star, 'fro')
    # print(f"Initial error: {initial_error}")
    U1, errors, iter = gradient_descent(U1, A, y, X_star,max_iter=1000)
    if errors[-1] < 1e-6:
        break
    else:
        U1 = np.concatenate((U1, np.random.randn(n,1)),axis=1)
        size = size + 1
        # U1 = np.random.randn(n, r)
    print(np.linalg.norm(A @ (U1 @ U1.T).flatten() - y) ** 2)
end = time.time()

# final error from X_star 
final_error = np.linalg.norm(U1 @ U1.T - X_star, 'fro')
print(f"Final error (heuristic): {final_error}")

# time and iteration
print(f"Time elapsed (heuristic): {end - start}")
print(f"Iteration (heuristic): {iter}")


U2 = np.random.randn(n, r)
start = time.time()
U2, errors, iter = gradient_descent(U2, A, y, X_star,max_iter=10000)
end = time.time()

final_error = np.linalg.norm(U2 @ U2.T - X_star, 'fro')
print(f"Final error (direct): {final_error}")

# time and iteration
print(f"Time elapsed (direct): {end - start}")
print(f"Iteration (direct): {iter}")

# verify gradient norm

# grad_U, grad_V = gradient_reg(U_reg, V_reg, A, y)
# print(f"Gradient U norm: {np.linalg.norm(grad_U)}")
# print(f"Gradient V norm: {np.linalg.norm(grad_V)}")

# plt.plot(errors)
# plt.xlabel('Iteration')
# plt.ylabel('Objective Function Value')
# plt.title('Convergence Plot Regularization')
# plt.show()