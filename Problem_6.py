# Question 6
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the ODE function
def ode(t, x, g):
    return -g

# Shooting algorithm
def shooting(g, t1, tol=1e-6, max_iter=100):
    # Define the initial and final times
    t0 = 0
    tf = t1
    # Set initial guess for x(t0)
    x0_guess = 0
    # Set initial guess for x'(t0)
    xp0_guess = 1
    # Iterate until convergence or max iterations reached
    for _ in range(max_iter):
        # Solve the ODE with the initial conditions
        sol = solve_ivp(ode, (t0, tf), [x0_guess], args=(g,), t_eval=[tf], method='RK45')
        # Get the solution at t1
        x1 = sol.y[0][-1]
        # Check convergence
        if np.abs(x1 - 0) < tol:
            return sol.y[0]
        # Adjust the initial condition using bisection method
        elif x1 < 0:
            x0_guess += tol
        else:
            x0_guess -= tol
    # If max iterations reached without convergence, return None
    return None

# Define parameters
g = 10
t1 = 10

# Generate candidate initial conditions
candidate_initial_conditions = np.linspace(-10, 10, 5)

# Solve the ODE for each candidate
numerical_solutions = []
for x0_guess in candidate_initial_conditions:
    sol = shooting(g, t1)
    if sol is not None:
        numerical_solutions.append(sol)

# Plot the exact solution, numerical solution, and candidate solutions
t_values = np.linspace(0, t1, 100)
exact_solution = 0.5 * g * t_values ** 2

#plt.plot(t_values, exact_solution, label='Exact Solution')
for solution in numerical_solutions:
    plt.plot(t_values, solution, label='Numerical Solution', linestyle='--')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.title('Shooting Method for Boundary Value Problem')
plt.legend()
plt.grid(True)
plt.show()