#Question_11
import numpy as np
import matplotlib.pyplot as plt

def ode(t, x):
    return 1 / (x**2 + t**2)

def rk4_step(t, x, h):
    k1 = h * ode(t, x)
    k2 = h * ode(t + h/2, x + k1/2)
    k3 = h * ode(t + h/2, x + k2/2)
    k4 = h * ode(t + h, x + k3)
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6

def adaptive_rk4(t0, x0, t_end, abs_tol):
    t_values = [t0]
    x_values = [x0]
    t = t0
    x = x0
    h = 0.01  # Initial step size
    
    while t < t_end:
        # Perform a single step with the current step size
        x_next = rk4_step(t, x, h)
        x_next_half = rk4_step(t, x, h/2)
        x_next_full = rk4_step(t + h/2, x_next_half, h/2)
        
        # Calculate error
        error = np.abs(x_next_full - x_next)
        max_error = np.max(error)
        
        # Adjust step size based on error
        h *= 0.9 * (abs_tol / max_error)**0.25
        
        # If error is within tolerance, accept the step
        if max_error < abs_tol:
            t += h
            x = x_next
            t_values.append(t)
            x_values.append(x)
    
    return np.array(t_values), np.array(x_values)

# Initial conditions
t0 = 0
x0 = 1

# Solve the ODE using adaptive step size RK4
t_values, x_values = adaptive_rk4(t0, x0, 3.5e6, 1e-14)

# Plot the solution
plt.plot(t_values, x_values, label='Numerical solution')
plt.xlabel('t')
plt.ylabel('x')
plt.xlim(0,100)
plt.title('Numerical Solution of dx/dt = 1/(x^2 + t^2) with Adaptive Step Size Control')
plt.grid(True)
plt.legend()
plt.show()

# Evaluate the solution at t = 3.5e6
t_eval = 3.5e6
x_eval = np.interp(t_eval, t_values, x_values)
print("Value of the solution at t =", t_eval, ":", x_eval)
