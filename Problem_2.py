# Question_2
import matplotlib.pyplot as plt
import numpy as np
def f(y,t):
    return y/t -(y/t)**2
def euler(f, y0, t):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    h = t[1] - t[0]
    for i in range(1, n):
        y_guess = y[i-1] + h * f(y[i-1], t[i-1])
        y[i] = y[i-1] + h * f(y_guess, t[i])
    return y
y1=1 #initial condition y(1)=1
t=np.linspace(1,2,10)
y_euler = euler(f, y1, t)  # solution using euler method

#actual solution-------------------
y_actual=t/(1+np.log(t))

#Error analysis------------------
A=y_actual-y_euler
Absoulute_error=abs(np.sum(A)/len(A))
print("Absoulute error : ",Absoulute_error)
B=A/y_actual
Relative_error=abs(np.sum(B)/len(B))
print("Relative error : ",Relative_error)
# Plot the solution--------------
plt.plot(t, y_euler, label=' Euler method solution with step size 0.1')
plt.plot(t, y_actual, label=' Analytic solution')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Comparison between the solutions ')
plt.legend()
plt.grid()
plt.show()


