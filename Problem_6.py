#Question 6
import numpy as np
import matplotlib.pyplot as plt


#Defining the Euler method of solving an ODE *****************
def euler(fx,fy,iny,inx,h,N):
    y = [iny]
    x = [inx]    
    for i in range(N):
        y.append(y[i]+h*fy())
        x.append(x[i]+h*fx(y[i]))
    return x


def fx(y):
    return y
def fy():
    return -10

x_0 = 0
x_n = 0
t_min = 0
t_max = 10
h = 0.01

N = int((t_max - t_min)/h)

g = 1
tol = 0.01
s0 = 0
s1 = 1

x = [euler(fx,fy,s0,x_0,h,N)]
s = [s0]


#Finding root by Secant method *******************************
while(np.abs(s1-s0)>tol):
    x0 = euler(fx,fy,s0,x_0,h,N)
    x1 = euler(fx,fy,s1,x_0,h,N)
    x.append(x1)
    s.append(s1)
    g0 = x_n - x0[-1]
    g1 = x_n -x1[-1]
    s2 = s1 - g1*((s1-s0)/(g1-g0))
    s0 = s1
    s1 = s2

t = [t_min]
for i in range(N):
    t.append(t_min+((i+1)*h))

print("The number of iterations before reaching the tolerence level = ",len(x))
print("For the two extra runs choosing the value of the first derivative to be 2 and 3.")

def real(t):
    sol = []
    for l in range(len(t)):
        sol.append(-5*t[l]**2 + 50*t[l])
    return sol
#print(real(t))


for a in range(len(x)):
    plt.plot(t,x[a],label='s = '+str(s[a]))
plt.plot(t,euler(fx,fy,2,x_0,h,N),label='s = 2')
plt.plot(t,euler(fx,fy,3,x_0,h,N),label='s = 3')
plt.plot(t,real(t),'--',label='Exact')
plt.xlabel('t')
plt.grid()
plt.ylabel('x')

plt.legend()
plt.show()




