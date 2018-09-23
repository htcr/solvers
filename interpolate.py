import numpy as np
import time
import matplotlib.pyplot as plt

class NewtonPoly(object):
    def __init__(self, coeffs, xs):
        self.coeffs = coeffs
        self.xs = xs
    
    def __call__(self, x):
        ans = 0
        last_term = 1
        for idx, a in enumerate(self.coeffs):
            ans += a*last_term
            last_term *= (x-self.xs[idx])
        return ans

def compute_div_diff(xs, fxs):
    ans = list()
    # column entry format: (f[xi...xj], xi, xj) 
    # first column
    col = [(fx, xs[idx], xs[idx]) for idx, fx in enumerate(fxs)]
    ans.append(col)
    n = len(xs)
    for i in range(1, n):
        col = list()
        for j in range(n-i):
            beg = ans[i-1][j][1]
            end = ans[i-1][j+1][2]
            top = ans[i-1][j][0] - ans[i-1][j+1][0]
            bottom = beg - end
            col.append((top/bottom, beg, end))
        ans.append(col)
    return ans

def interpolate(xs, fxs):
    div_diff = compute_div_diff(xs, fxs)
    coeffs = [col[0][0] for col in div_diff]
    return NewtonPoly(coeffs, xs)


if __name__ == '__main__':
    # 1a
    print('---------1a----------')
    func_1a = lambda x: (np.log(x)/np.log(6))**(1.5)
    xs = np.arange(1.0, 4.1, 0.5)
    fxs = func_1a(xs)
    func_1a_approx = interpolate(xs, fxs)
    val_x = 2.25
    print('interpolated f(%f)=%f' % (val_x, func_1a_approx(val_x)))
    print('actuall f(%f)=%f' % (val_x, func_1a(val_x)))

    # 1b
    print('---------1b----------')
    def estimate(func, x, n):
        xs = np.array([2.0*i/n - 1 for i in range(0, n+1)])
        fxs = func(xs)
        func_approx = interpolate(xs, fxs)
        approx_fx = func_approx(x)
        actual_fx = func(x)
        print('x: %f, n: %d, approx: %f, actual: %f' % (x, n, approx_fx, actual_fx))

    func_1b = lambda x: 6.0/(1+25*(x**2))
    x = 0.05
    ns = [2, 4, 40]
    for n in ns:
        estimate(func_1b, 0.05, n)

    # 1c
    print('---------1b----------')
    ns = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40]
    for n in ns:
        xs = np.array([2.0*i/n - 1 for i in range(0, n+1)])
        fxs = func_1b(xs)
        func_1b_approx = interpolate(xs, fxs)
        eexs = np.arange(-1, 1.0001, 0.01)
        feexs = func_1b(eexs)
        feexs_approx = func_1b_approx(eexs)

        # visualize
        # plt.plot(eexs, feexs, 'b')
        # plt.plot(eexs, feexs_approx, 'r')
        # plt.show()

        max_error = np.max(np.abs(feexs_approx-feexs))
        argmax_error = np.argmax(np.abs(feexs_approx-feexs))
        # print('argmax error: x=%f' % (eexs[argmax_error]))
        print('n: %d, max_error: %f' % (n, max_error))