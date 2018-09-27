import numpy as np
from interpolate import compute_div_diff
from newton import newton

eps = 0.01

class Poly(object):
    def __init__(self, coeffs):
        # first element of coeffs is always for constant term.
        self.coeffs = np.array(coeffs, dtype=np.complex128)
        self.powers = np.arange(self.coeffs.shape[0], dtype=np.int32)
        self.degree = self.coeffs.shape[0] - 1

    def __call__(self, x):
        return np.sum((x**self.powers)*self.coeffs)
    
    def __repr__(self):
        return str(self.coeffs)


def deflate(p, r):
    remnant = np.array(p.coeffs, dtype=np.complex128)
    deflated = np.zeros(p.coeffs.shape, dtype=np.complex128)
    buff = np.zeros(p.coeffs.shape, dtype=np.complex128)
    for i in range(deflated.shape[0]-2, -1, -1):
        buff.fill(0)
        buff[i:i+2] = (-r, 1)
        deflated[i] = remnant[i+1]
        remnant -= (deflated[i]*buff)
    remnant_mag = np.max(np.abs(remnant))
    assert remnant_mag < eps, 'illegal deflation, remnant: %s' % (str(remnant))
    return Poly(deflated[:-1])


def muller_one_iter(f, x0, x1, x2):
    fx0, fx1, fx2 = f(x0), f(x1), f(x2)
    div_diff = compute_div_diff([x0, x1, x2], [fx0, fx1, fx2])
    c = fx2
    a = div_diff[2][0][0]
    b = div_diff[1][1][0] + a * (x2 - x1)
    
    root_item = np.sqrt(b**2 - 4*a*c)
    d0 = b + root_item
    d1 = b - root_item
    if np.abs(d0) >= np.abs(d1):
        x3 = x2 - 2*c / d0
    else:
        x3 = x2 - 2*c / d1
    
    return x1, x2, x3


def muller(f, n, x0, x1, x2):
    for i in range(n):
        x0, x1, x2 = muller_one_iter(f, x0, x1, x2)
    return x2

def muller_init_poly(p):
    r = 1 + np.max(np.abs(p.coeffs[:-1]/p.coeffs[-1]))
    return -r, 0, r

def diff_poly(p):
    return Poly(p.coeffs[1:]*p.powers[1:])

def solve_poly(p):
    roots = list()
    cur_p = p
    diff_p = diff_poly(p)
    for i in range(p.degree):
        if len(roots) > 0:
            cur_p = deflate(cur_p, roots[-1])

        x0, x1, x2 = muller_init_poly(cur_p)
        r = muller(cur_p, 2, x0, x1, x2)
        r = newton(r, p, diff_p, 10)
        roots.append(r)
    return roots



p = Poly([-4, 6, -4, 1])

#p = Poly([-1, -12, 0, 16, 16])

# print(diff_poly(p))

roots = solve_poly(p)
for idx, r in enumerate(roots):
    print('root %d:' % (idx))
    print(r)
    print('function value:')
    print(p(r))
