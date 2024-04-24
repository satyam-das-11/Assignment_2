#Question_14
import numpy as np
import matplotlib.pyplot as plt

def second_order_ode(t, y):
    # Extract y and y'
    y1, y2 = y
    
    # Compute derivatives
    dy1dt = y2
    dy2dt = t*np.log(t)+2.0*y2/t-2.0*y1/t**2
    
    return [dy1dt, dy2dt]

def euler_method(func, y0, t):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    h = t[1] - t[0]

    for i in range(1, n):
        y[i] = y[i-1] + h * np.array(func(t[i-1], y[i-1]))

    return y

# Initial conditions
y0 = [1.0, 0.0]  # Initial displacement and velocity
t = np.linspace(1, 2, 1000)  # Time points

# Solve the ODE using Euler method
solution = euler_method(second_order_ode, y0, t)
# Extract displacement and velocity
y= solution[:, 0]
dydt = solution[:, 1]

def Analytic(t):
    return 7.0*t/4.0 +(t**3/2.0)*np.log(t)-(3.0/4)*t**3
x=np.linspace(1,2,1000)
y_analytic=Analytic(x)
# Plot the results
plt.plot(t, y, label='Euler solution')
plt.plot(x, y_analytic, label='Analytic solution')
plt.xlabel('Time')
plt.ylabel('Displacement / Velocity')
plt.title('Solution of Second Order ODE using Euler Method')
plt.legend()
plt.grid(True)
plt.show()