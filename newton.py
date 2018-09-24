import numpy as np

f = lambda x:np.tan(x) - x
df = lambda x: 1.0 / ((np.cos(x))**2) - 1

def newton(x0, f, df, n):
    x = x0
    for i in range(n):
        x = x - f(x)/df(x)
    return x


if __name__ == '__main__':
    root_left = newton(10.904, f, df, 5)
    print('left root newton: %f' % root_left)
    print(f(root_left))

    root_right = newton(14.066, f, df, 5)
    print('right root newton: %f' % root_right)
    print(f(root_right))