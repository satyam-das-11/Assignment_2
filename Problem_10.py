# Question 10
import numpy as np
import matplotlib.pyplot as plt

def ode(t, y):
    return (y**2 + y) / t

def rk4_step(t, y, h):
    k1 = h * ode(t, y)
    k2 = h * ode(t + h/2, y + k1/2)
    k3 = h * ode(t + h/2, y + k2/2)
    k4 = h * ode(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def adaptive_rk4(t0, y0, t_end, h_init, abs_tol):
    t_values = [t0]
    y_values = [y0]
    h = h_init
    t = t0
    
    while t < t_end:
        y_next = rk4_step(t, y_values[-1], h)
        y_next_half = rk4_step(t, y_values[-1], h/2)
        y_next_full = rk4_step(t + h/2, y_next_half, h/2)
        
        error = np.abs(y_next_full - y_next)
        max_error = np.max(error)
        
        if max_error < abs_tol:
            y_values.append(y_next_full)
            t += h
            t_values.append(t)
        h = 0.9 * h * (abs_tol / max_error)**0.25
    
    return np.array(t_values), np.array(y_values)

# Initial conditions
t0 = 1
y0 = -2

# Solve the ODE using adaptive step-size RK4
t_values, y_values = adaptive_rk4(t0, y0, 3, 0.1, 1e-4)

# Plot the solution and mesh points
plt.plot(t_values, y_values, marker='o', label='Numerical solution')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of y\' = (y^2 + y) / t')
plt.grid(True)
plt.legend()
plt.show()