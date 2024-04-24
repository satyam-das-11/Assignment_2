import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#1st ODE

# Define the first-order ODE function
def f1(t, y):
    return t * np.exp(3 * t) - 2 * y

# Define the initial condition
y0_1 = [0]
t_span_1 = (0, 1)

# Solve the initial value problem using solve_ivp
sol_1 = solve_ivp(f1, t_span_1, y0_1,dense_output=True)
#Analytic solution----------------
x_1=np.linspace(0,1,80)
def Analytic_f1(x):
    return (1.0/25)*np.exp(-2*x)*(1-np.exp(5*x)+5*x*np.exp(5*x))

# Evaluate the solution at desired time points
t_eval_1 = np.linspace(0, 1, 10)
stepsize_1=(max(t_eval_1)-min(t_eval_1))/float(len(t_eval_1))
y_eval_1 = sol_1.sol(t_eval_1).flatten()
y_analytic_1=Analytic_f1(x_1)
# Plot the solution
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#2nd ODE
def f2(t, y):
    return 1-(t-y)**2

y0_2 = [1]
t_span_2 = (2, 3)

# Solve the initial value problem using solve_ivp
sol_2 = solve_ivp(f2, t_span_2, y0_2,dense_output=True)
#Analytic solution----------------
x_2=np.linspace(2,2.99,100)
def Analytic_f2(x):
    return (1.0-3*x+ x**2)/(-3.0+x)

t_eval_2 = np.linspace(2, 2.99, 40)
stepsize_2=(max(t_eval_2)-min(t_eval_2))/float(len(t_eval_2))
y_eval_2 = sol_2.sol(t_eval_2).flatten()
y_analytic_2=Analytic_f2(x_2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#3rd ODE
def f3(t, y):
    return 1+y/t
y0_3 = [2]
t_span_3 = (1,2)

sol_3 = solve_ivp(f3, t_span_3, y0_3,dense_output=True)
#Analytic solution----------------
x_3=np.linspace(1,2,100)
def Analytic_f3(x):
    return 2*x+x*np.log(x)

# Evaluate the solution at desired time points
t_eval_3 = np.linspace(1, 2, 10)
stepsize_3=(max(t_eval_3)-min(t_eval_3))/float(len(t_eval_3))
y_eval_3 = sol_3.sol(t_eval_3).flatten()
y_analytic_3=Analytic_f3(x_3)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#4th ODE
def f4(t, y):
    return np.cos(2*t)+np.sin(3*t)
y0_4 = [1]
t_span_4 = (0,1)

sol_4 = solve_ivp(f4, t_span_4, y0_4,dense_output=True)
#Analytic solution----------------
x_4=np.linspace(0,1,100)
def Analytic_f4(x):
    return (1.0/6)*(8-2*np.cos(3*x)+3*np.sin(2*x))

# Evaluate the solution at desired time points
t_eval_4 = np.linspace(0,1, 10)
stepsize_4=(max(t_eval_4)-min(t_eval_4))/float(len(t_eval_4))
y_eval_4 = sol_4.sol(t_eval_4).flatten()
y_analytic_4=Analytic_f4(x_4)

#plotting---------------------------------------------------------
plt.subplot(221)
plt.plot(t_eval_1, y_eval_1,label="numerical")
plt.plot(x_1,y_analytic_1,label="Analytic")
plt.xlabel('t')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.title(r'$\frac{dy}{dt} $ = $te^{3t} -2y$ , stepsize h=%f'%stepsize_1)
plt.subplot(222)
plt.plot(t_eval_2, y_eval_2,label="numerical")
plt.plot(x_2,y_analytic_2,label="Analytic")
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.title(r'$\frac{dy}{dt} $ = 1-$(t-y)^2$ , stepsize h=%f'%stepsize_2)
plt.subplot(223)
plt.plot(t_eval_3, y_eval_3,label="numerical")
plt.plot(x_3,y_analytic_3,label="Analytic")
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.title(r'$\frac{dy}{dt} $ = $1+\frac{y}{t}$ , stepsize h=%f'%stepsize_3)
plt.subplot(224)
plt.plot(t_eval_4, y_eval_4,label="numerical")
plt.plot(x_4,y_analytic_4,label="Analytic")
plt.xlabel('t')
plt.ylabel('y')
plt.title(r'$\frac{dy}{dt} $ = $\cos(2t)+\sin(3t)$ , stepsize h=%f'%stepsize_4)
plt.grid()
plt.legend()
plt.show()