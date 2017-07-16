import numpy as np
import math

def getMFHartmann(numFidelities, dim):
    alpha = np.array([1., 1.2, 3., 3.2])
    delta = np.array([0.01, -0.01, -0.1, 0.1])
    if dim == 3:
        fn = hartmann3
    if dim == 6:
        fn = hartmann6
        delta *= 0.1
    return lambda x, f: -fn(x, alpha + delta * (numFidelities - f - 1))

def hartmann(x, A, P, alpha):

    result = 0.
    for i in range(4):
        tmp = 0.
        for j in range(len(x)):
            tmp -= A[i][j] * (x[j] - P[i][j]) ** 2.;
        result -= alpha[i] * math.exp(tmp)
    return result
'''
http://www.sfu.ca/~ssurjano/hart3.html
Domain: (0, 1)^3
Gloabal Minimum: f(x) = -3.86278
    for x = (0.114614, 0.555649, 0.852547)
'''
def hartmann3(x, alpha):

    A = [[3., 10., 30.],
        [0.1, 10., 35.],
        [3., 10., 30.],
        [0.1, 10., 35.]]

    P = [[.3689, .1170, .2673],
        [.4699, .4387, .7470],
        [.1091, .8732, .5547],
        [.0381, .5743, .8828]]

    return hartmann(x, A, P, alpha)

'''
http://www.sfu.ca/~ssurjano/hart6.html
The 6-dimensional Hartmann function has 6 local minima.
Domain: (0, 1)^6
Global Minimum: f(x) = -3.32237 (Made negative to be a maximum)
    for x = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
'''
def hartmann6(x, alpha):

    A = [[10., 3., 17., 3.5, 1.7, 8.],
        [0.05, 10., 17., 0.1, 8., 14.],
        [3., 3.5, 1.7, 10., 17., 8.],
        [17., 8., 0.05, 10., 0.1, 14.]]

    P = [[.1312, .1696, .5569, .0124, .8283, .5886],
        [.2329, .4135, .8307, .3736, .1004, .9991],
        [.2348, .1451, .3522, .2883, .3047, .6650],
        [.4047, .8828, .8732, .5743, .1091, .0381]]

    return hartmann(x, A, P, alpha)

'''
https://www.sfu.ca/~ssurjano/park91a.html
According to Mathematica:
Global Max: f(x) = 25.5893 where x = (1, 1, 1, 1) on [0, 1]^4.
'''
def park1(x):
    x1, x2, x3, x4 = x

    a = math.sqrt(1. + (x2 + x3**2.) * x4 / x1**2.) - 1.
    b = (x1 + 3.*x4) * math.exp(1. + math.sin(x3))
    return 0.5*x1 * a + b

def lowFidelityPark1(x):
    x1, x2, x3, _ = x

    p = -2.*x1 + x2**2. + x3**2. + 0.5
    return (1. + math.sin(x1)/10.) * park1(x) + p

'''
https://www.sfu.ca/~ssurjano/park91b.html
According to Mathematica:
Global Max: f(x) = 5.92604 where x = (1, 1, 1, 0) on [0, 1]^4.
'''
def park2(x):
    x1, x2, x3, x4 = x

    return 2./3. * math.exp(x1 + x2) - x4 * math.sin(x3) + x3

def lowFidelityPark2(x):
    return 1.2 * park2(x) - 1.

'''
https://www.sfu.ca/~ssurjano/curretal88exp.html
According to Mathematica:
Global Max: f(x) = 13.7987 where x = (0.216667, 0.0228407) on [0, 1]^2.
'''
def currinExponential(args):
    x, y = args

    p = 2300. * x ** 3. + 1900. * x ** 2. + 2092. * x + 60.
    q = 100. * x ** 3. + 500. * x ** 2. + 4. * x + 20
    if y == 0:
        r = 0
    else:
        r = math.exp(-.5 / y)

    return (1. - r) * p / q

def lowFideliltyCurrinExponential(args):
    x, y = args
    a = (x + .05, y + .05)
    b = (x + .05, max(0, y - .05))
    c = (x - .05, y + .05)
    d = (x - .05, max(0, y - .05))

    return .25 * (currinExponential(a) + currinExponential(b)
                + currinExponential(c) + currinExponential(d))

'''
https://www.sfu.ca/~ssurjano/borehole.html
Global max: 309.523221
Bounds: [0.05 0.15; ...
         100, 50000; ...
         63070, 115600; ...
         990, 1110; ...
         63.1, 116; ...
         700, 820; ...
         1120, 1680; ...
         9855, 12045];
'''
def borehole(x):
    (rw, r, Tu, Hu, T1, H1, L, Kw) = x
    frac1 = 2. * math.pi * Tu * (Hu - H1)
    frac2a = 2. * L * Tu / (math.log(r / rw) * rw ** 2. * Kw)
    frac2b = Tu / T1
    frac2 = math.log(r / rw) * (1. + frac2a + frac2b)
    return frac1 / frac2

def lowFidelityBorehole(x):
    (rw, r, Tu, Hu, T1, H1, L, Kw) = x
    frac1 = 5. * Tu * (Hu - H1)
    frac2a = 2. * L * Tu / (math.log(r / rw) * rw ** 2. * Kw)
    frac2b = Tu / T1
    frac2 = math.log(r / rw) * (1.5 + frac2a + frac2b);
    return frac1 / frac2;
