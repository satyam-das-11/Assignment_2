#Question 7 
import numpy as np
import matplotlib.pyplot as plt



xi = 0
xf = 0
ti = 0
tf = 10
N = 100
x = np.zeros(N+1)
x[0],x[-1] = xi,xf
h = (tf-ti)/N
a = 1
tol = 10**(-1)

t = np.linspace(ti,tf,N+1)
q = 0
while(a>tol):
    if (q%200==0):  #To show iterations whihch are multiple of 200
        plt.plot(t,x,label = 'Iteration no.'+str(q))
    q = q+1 
    y = np.copy(x)
    for i in range(1,len(x)-1):
        x[i] = (x[i-1]+x[i+1]+10*h**2)/2
    a = np.linalg.norm(x-y) #considering L2 norm

plt.plot(t,x,label = 'Final solution')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.grid()
plt.show()