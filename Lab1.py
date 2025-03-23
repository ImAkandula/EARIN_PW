import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def function(x, y):
    return 2 * np.sin(x) + 3 * np.cos(y)

def gradient(x, y):
    df_dx = 2 * np.cos(x)
    df_dy = -3 * np.sin(y)
    return np.array([df_dx, df_dy])

def hessian(x, y):
    d2f_dx2 = -2 * np.sin(x)
    d2f_dy2 = -3 * np.cos(y)
    return np.array([[d2f_dx2, 0],
                     [0, d2f_dy2]])

def newton_method(initial_guess, alpha, tol=1e-6, max_iter=1000):
    x = np.array(initial_guess, dtype=float)
    num_iter = 0
    for i in range(max_iter):
        grad = gradient(x[0], x[1])
        hess = hessian(x[0], x[1])
        if np.linalg.det(hess) == 0:
            print("Cannot be solved as Hess inverse doen't exist", i)
            break
        delta = np.linalg.solve(hess, grad)
        x_new = x - alpha * delta
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
        num_iter += 1
    return x, num_iter

def visualize():
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = function(X, Y)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none')
    ax.set_title('Function Surface: 2sin(x) + 3cos(y)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

visualize()

test_cases = [
    ([2.0, 2.0], 0.1),
    ([-4.0, -4.0], 0.05),
    ([1.0, -3.0], 0.2),
    ([-2.0, 4.0], 0.15),
    ([0.5, 0.5], 0.1),
    ([0, 5], 0.1) #hessian inverse doesn't exist
]

for initial_guess, learning_rate in test_cases:
    minimum, iterations = newton_method(initial_guess, learning_rate)
    print(f"Initial guess: {initial_guess}, Learning rate: {learning_rate}")
    print(f"Minimum approximation: {minimum}, Iterations: {iterations}\n")
