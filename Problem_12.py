#Question_12
import numpy as np
import matplotlib.pyplot as plt

def ode(u, t):
    u1,u2,u3=u
    du1dt=u1+2*u2-2*u3+np.exp(-t)
    du2dt=u2+u3-2*np.exp(-t)
    du3dt=u1+2*u2+np.exp(-t)  
    return [du1dt,du2dt,du3dt]

def rk4_method(func, y0, t):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    h = t[1] - t[0]  # step size

    for i in range(1, n):
        k1 = h * np.array(func(y[i-1], t[i-1]))
        k2 = h * np.array(func(y[i-1] + 0.5 * k1, t[i-1] + 0.5 * h))
        k3 = h * np.array(func(y[i-1] + 0.5 * k2, t[i-1] + 0.5 * h))
        k4 = h * np.array(func(y[i-1] + k3, t[i-1] + h))
        y[i] = y[i-1] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    return y

# Initial conditions
u0 = [3.0, -1.0,1.0]  # initial u1,u2,u3
t = np.linspace(0, 1, 100)  # t points

# Solve the ODE using RK4 method
solution = rk4_method(ode, u0, t)

# Extract u1,u2,u3
u1 = solution[:, 0]
u2 = solution[:, 1]
u3 = solution[:, 2]


# Plot the results
plt.plot(t, u1, label='u1')
plt.plot(t, u2, label='u2')
plt.plot(t, u3, label='u3')
plt.xlabel('t')
plt.ylabel('u1 / u2 / u3')
plt.title('Solution of a family of ODEs using RK4 Method')
plt.legend()
plt.grid()
plt.show()