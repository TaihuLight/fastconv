import numpy as np
from numpy.fft import fft
from numpy.fft import ifft
from scipy.signal import convolve

# linear convolution, padding with p = k-1
def LinearConv(x, k):
    return convolve(x, k, 'full')

# valid convolution, also no padding
def CircularConv(x, k):
    return convolve(x, k, 'valid')

# We need zero padding the signal filter ending,
# and the length must be at least L + M -1.
# extpad means extra padding at the end of signal and filter
def FFTLinearConv(x, k, extpad = 0):
    assert len(x.shape) == 1
    assert len(k.shape) == 1
    assert extpad >= 0

    L = x.shape[0]
    M = k.shape[0]
    T0 = L + M - 1
    T = T0 + extpad
    return np.real(ifft(fft(x, T) * fft(k, T)))[0:T0]

# valid convolution donot need padding signal
# extpad means extra padding at the end of signal and filter
def FFTCircularConv(x, k, extpad = 0):
    assert len(x.shape) == 1
    assert len(k.shape) == 1
    assert extpad >= 0

    L = x.shape[0]
    M = k.shape[0]
    T = L + extpad
    T0 = L - M + 1
    return np.real(ifft(fft(x, T) * fft(k, T)))[M-1: M-1+T0]


# check the result
def CheckResult(x, y, tol = 1e-4):
    assert x.shape == y.shape
    assert x.size == y.size
    if np.max(np.abs(x-y)) <= tol:
        return True
    else:
        return False

if __name__ == "__main__":
    L = 10
    M = 5
    x = np.random.random(L)
    k = np.random.random(M)
    conv1 = LinearConv(x, k)
    conv2 = FFTLinearConv(x, k, 3)
    conv3 = CircularConv(x, k)
    conv4 = FFTCircularConv(x, k, 4)

    print("signal:\n", x)
    print("filter:\n", k)
    print("LinearConv:\n", conv1)
    print("FFTLinearConv:\n", conv2)
    print("CircularConv:\n", conv3)
    print("FFTCircularConv:\n", conv4)

    if CheckResult(conv1, conv2):
        print("Check LinearConv success.")
    else:
        print("Check LinearConv fail!")

    if CheckResult(conv3, conv4):
        print("Check CircularConv success.")
    else:
        print("Check FFTCircularConv fail!")

