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

def newton_method_with_tracking(initial_guess, alpha, tol=1e-6, max_iter=1000):
    x = np.array(initial_guess, dtype=float)
    path = [x.copy()]
    num_iter = 0
    for i in range(max_iter):
        grad = gradient(x[0], x[1])
        hess = hessian(x[0], x[1])
        if np.linalg.det(hess) == 0:
            print(f"Hessian is not invertible at iteration {i} for initial {initial_guess}")
            return None, num_iter, np.array(path)
        delta = np.linalg.solve(hess, grad)
        x_new = x - alpha * delta
        path.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
        num_iter += 1
    return x, num_iter, np.array(path)

def visualize(results):
    X = np.linspace(-5, 5, 150)
    Y = np.linspace(-5, 5, 150)
    X, Y = np.meshgrid(X, Y)
    Z = function(X, Y)

    
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_axes([0.05, 0.1, 0.65, 0.8], projection='3d')  
    ax_text = fig.add_axes([0.72, 0.1, 0.25, 0.8])  
    ax_text.axis('off')

    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, edgecolor='none', alpha=0.3)

    
    zero_plane = np.zeros_like(Z)
    ax.plot_surface(X, Y, zero_plane, color='gray', alpha=0.2)

    
    path_colors = ['blue', 'green', 'orange', 'purple', 'crimson', 'magenta', 'darkcyan', 'darkred', 'darkblue']

    
    details_text = "Newton's Method Details:\n"
    details_text += "{:<10} {:<20} {:<25} {}\n".format("Test", "Iterations", "Converged (x, y)", "Z-Value")

    for idx, result in enumerate(results):
        path = result['path']
        initial = path[0]
        final = path[-1]
        z_path = function(path[:, 0], path[:, 1])

        
        ax.plot(path[:, 0], path[:, 1], z_path,
                color=path_colors[idx % len(path_colors)],
                linewidth=2.5,
                label=f"Path {idx+1}")

        
        ax.scatter(initial[0], initial[1], function(initial[0], initial[1]),
                   color='black', s=80, marker='o', label=f"Start {idx+1}")

        
        ax.scatter(path[:, 0], path[:, 1], z_path,
                   color=path_colors[idx % len(path_colors)], s=40, alpha=0.8)

        
        ax.scatter(final[0], final[1], function(final[0], final[1]),
                   color='yellow', edgecolor='black', s=120, marker='X', label=f"Converged {idx+1}")

        
        details_text += "{:<10} {:<20} ({: .3f}, {: .3f})       {: .3f}\n".format(
            idx+1, result['iterations'], final[0], final[1], function(final[0], final[1])
        )

    ax.set_title('Newton\'s Method Iteration Paths and Convergence', fontsize=14)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    
    ax_text.text(0, 1, details_text, fontsize=10, verticalalignment='top', family='monospace')

    plt.show()


test_cases = [
    ([2.0, 2.0], 0.1),
    ([-4.0, -4.0], 0.05),
    ([1.0, -3.0], 0.2),
    ([-2.0, 4.0], 0.15),
    ([0.5, 0.5], 0.1),
    ([0, 5], 0.1),
    ([4.5, -4.5], 0.1),
    ([-3, 3], 0.1),
    ([-3, 3], 0.5),
    ([-3, 3], 1)
]

# Run Newton's method and collect results
results = []
for initial_guess, learning_rate in test_cases:
    minimum, iterations, path = newton_method_with_tracking(initial_guess, learning_rate)
    results.append({
        'initial': initial_guess,
        'minimum': minimum,
        'iterations': iterations,
        'path': path
    })



# Print test cases and results in the console
print("\nNewton's Method Results Summary:\n")
print("{:<10} {:<25} {:<15} {:<30} {}".format("Test", "Initial Guess (x, y)", "Iterations", "Converged (x, y)", "Z-Value"))

for idx, result in enumerate(results):
    initial = result['initial']
    final = result['minimum']
    iterations = result['iterations']

    if final is None:
        print("{:<10} ({: .4f}, {: .4f})      {:<15} {:<30} {}".format(
            idx + 1, initial[0], initial[1], iterations,
            "Hessian not invertible", "N/A"))
    else:
        z_value = function(final[0], final[1])
        print("{:<10} ({: .4f}, {: .4f})      {:<15} ({: .4f}, {: .4f})      {: .4f}".format(
            idx + 1, initial[0], initial[1], iterations, final[0], final[1], z_value))



# Visualize everything with side details
visualize(results)