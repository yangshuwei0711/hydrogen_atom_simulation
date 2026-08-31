import numpy as np
import scipy.special as sp

def WaveFuncNorm(a, n, l):
    A = (2 / (n * a))**3
    B = sp.factorial(n - l - 1) / (2 * n * sp.factorial(n + l))
    N = np.sqrt(A * B)
    return N

def RadialWaveFunc(r, a, n, l):
    A = np.exp(-r / (n * a))
    B = (2 * r / (n * a))**l
    C = sp.assoc_laguerre(2 * r / (n * a), n - l - 1, 2 * l + 1)
    R = A * B * C
    return R

def SphericalHarmonic(theta, phi, l, m):
    Y = sp.sph_harm_y(l, m, theta, phi)
    return Y

def WaveFunction(r, theta, phi, a, n, l, m):
    N = WaveFuncNorm(a, n, l)
    R = RadialWaveFunc(r, a, n, l)
    Y = SphericalHarmonic(theta, phi, l, m)
    Psi = N * R * Y
    return Psi

def ProbabilityDensity(r, theta, phi, a, n, l, m):
    Psi = WaveFunction(r, theta, phi, a, n, l, m)
    P = np.abs(Psi)**2
    return P