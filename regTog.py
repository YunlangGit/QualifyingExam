import numpy as np
import matplotlib.pyplot as plt
import time

# generate operator A
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

# calculate the objective function with regularization
def calculate_g(U, V, A, y):
    UV_T = U @ V.T
    AUV = A @ (UV_T.flatten())
    residual = AUV - y
    A_term = np.sum(residual**2)
    regularization = 0.25 * np.linalg.norm(U.T @ U - V.T @ V, 'fro')**2

    return A_term + regularization

# calculate the gradient (without regularization)
def gradient(U, V, A, y):
    k = len(A)
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)

    UVT = U @ V.T
    for i in range(k):
        Ai = A[i]
        residual = Ai @ UVT.flatten() - y[i]
        grad_U += 2 * residual * Ai.reshape(m,n) @ V
        grad_V += 2 * residual * (Ai.reshape(m,n)).T @ U

    return grad_U / k, grad_V / k

# gradient descent without regularization term
def gradient_descent(U, V, A, y, X_star, max_iter=1000, tol=1e-6):
    alpha = 10
    eta = 0.1
    errors = [np.linalg.norm(A @ (U @ V.T).flatten() - y) ** 2]
    for i in range(max_iter):
        # update U
        f = np.linalg.norm(A @ (U @ V.T).flatten() - y) ** 2
        grad_U, grad_V = gradient(U, V, A, y)
        norm_grad_U = np.linalg.norm(grad_U, 'fro')
        Utemp = U - alpha * grad_U
        ftemp = np.linalg.norm(A @ (Utemp @ V.T).flatten() - y) ** 2
        while ftemp > f - eta * alpha * np.linalg.norm(grad_U)**2:
            alpha *= 0.5
            Utemp = U - alpha * grad_U
            ftemp = np.linalg.norm(A @ (Utemp @ V.T).flatten() - y) ** 2
        U = Utemp

        # update V
        f = np.linalg.norm(A @ (U @ V.T).flatten() - y) ** 2
        grad_U, grad_V = gradient(U, V, A, y)
        norm_grad_V = np.linalg.norm(grad_V, 'fro')
        Vtemp = V - alpha * grad_V
        ftemp = np.linalg.norm(A @ (U @ Vtemp.T).flatten() - y) ** 2
        while ftemp > f - eta * alpha * np.linalg.norm(grad_V)**2:
            alpha *= 0.5
            Vtemp = V - alpha * grad_V
            ftemp = np.linalg.norm(A @ (U @ Vtemp.T).flatten() - y) ** 2
        V = Vtemp

        error = np.linalg.norm(A @ (U @ Vtemp.T).flatten() - y) ** 2
        errors.append(error)
        if error < tol or (norm_grad_V+norm_grad_U) < tol:
            break
    return U, V, errors, i

# calcualte the gradient with the regularization
def gradient_reg(U, V, A, y):
    k = len(A)
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)

    UVT = U @ V.T
    for i in range(k):
        Ai = A[i]
        residual = Ai @ UVT.flatten() - y[i]
        grad_U += 2 * residual * Ai.reshape(m,n) @ V
        grad_V += 2 * residual * (Ai.reshape(m,n)).T @ U
    grad_U += (U @ U.T @ U - U @ V.T @ V)
    grad_V += (V @ V.T @ V - V @ U.T @ U)
    return grad_U / k, grad_V / k

# gradient descent with regularization term
def gradient_descent_reg(U, V, A, y, X_star, max_iter=1000, tol=1e-6):
    alpha = 10
    eta = 0.1
    errors = [calculate_g(U, V, A, y)]
    for i in range(max_iter):
        # update U
        f = calculate_g(U, V, A, y)
        grad_U, grad_V = gradient_reg(U, V, A, y)
        norm_grad_U = np.linalg.norm(grad_U, 'fro')
        Utemp = U - alpha * grad_U
        ftemp = calculate_g(Utemp, V, A, y)
        while ftemp > f - eta * alpha * np.linalg.norm(grad_U)**2:
            alpha *= 0.5
            Utemp = U - alpha * grad_U
            ftemp = calculate_g(Utemp, V, A, y)
        U = Utemp

        # update V
        f = calculate_g(U, V, A, y)
        grad_U, grad_V = gradient_reg(U, V, A, y)
        norm_grad_V = np.linalg.norm(grad_V, 'fro')
        Vtemp = V - alpha * grad_V
        ftemp = calculate_g(U, Vtemp, A, y)
        while ftemp > f - eta * alpha * np.linalg.norm(grad_V)**2:
            alpha *= 0.5
            Vtemp = V - alpha * grad_V
            ftemp = calculate_g(U, Vtemp, A, y)
        V = Vtemp

        error = calculate_g(U, V, A, y)
        errors.append(error)
        if error < tol or (norm_grad_V+norm_grad_U) < tol:
            break
    return U, V, errors, i

# parameters
m, n, r, k = 100, 50, 10, 4000

print('-----------Start-----------')

repeat = 5

for i in range(repeat):
    print("iteration",i)

    # synthesize data
    A = generate_A_matrices(m, n, k)
    X_star = generate_X(m, n, r)
    y = calculate_y(A, X_star)

    # initialize U and V
    U0 = np.random.randn(m, r)
    V0 = np.random.randn(n, r)

    # initial error from X_star
    initial_error = np.linalg.norm(U0 @ V0.T - X_star, 'fro')
    print(f"Initial error: {initial_error}")

    # run gradient descent without regularization
    start = time.time()
    U_nr, V_nr, errors, iter = gradient_descent(U0, V0, A, y, X_star)
    end = time.time()

    # final error from X_star
    final_error = np.linalg.norm(U_nr @ V_nr.T - X_star, 'fro')
    print(f"Final error (no regularization): {final_error}")

    # time and iteration
    print(f"Time elapsed (no regularization): {end - start}")
    print(f"Iteration (no regularization): {iter}")

    # verify gradient norm

    # grad_U_ur, grad_V_ur = gradient(U_nr, V_nr, A, y)
    # print(f"Gradient U norm: {np.linalg.norm(grad_U_ur)}")
    # print(f"Gradient V norm: {np.linalg.norm(grad_V_ur)}")

    # run gradient descent with regularization
    start = time.time()
    U_reg, V_reg, errors_reg, iter_reg = gradient_descent_reg(U0, V0, A, y, X_star)
    end = time.time()

    # final error from X_star 
    final_error = np.linalg.norm(U_reg @ V_reg.T - X_star, 'fro')
    print(f"Final error (regularization): {final_error}")

    # time and iteration
    print(f"Time elapsed (regularization): {end - start}")
    print(f"Iteration (regularization): {iter_reg}")

    # verify gradient norm

    # grad_U_reg, grad_V_reg = gradient(U_reg, V_reg, A, y)
    # print(f"Gradient U norm: {np.linalg.norm(grad_U_reg)}")
    # print(f"Gradient V norm: {np.linalg.norm(grad_V_reg)}")

    # error plot 
    plt.plot(errors)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Convergence Plot without Regularization')
    plt.show()

    plt.plot(errors_reg)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Convergence Plot Regularization')
    plt.show()