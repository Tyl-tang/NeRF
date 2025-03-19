import sympy as sp
from decimal import Decimal
def func_1(x):
    x = x
    return 0.01*x**2

def n_diff(f,x):
    h = 0.0001

    m = f(x+h)
    l = f(x-h)
    k= m-l
    n = Decimal(2*h)
    return k/n

y = n_diff(func_1,5)
print(y)