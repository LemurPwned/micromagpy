import numpy as np
from scipy.fft import fft, ifft


def alpha_noise(N, Qstd, alpha):
    white_noise = np.random.normal(0, Qstd, N)
    # white_noise = np.ones_like(white_noise)
    hnoise = np.zeros_like(white_noise)
    hnoise[0] = 1.0
    for i in range(1, N):
        hnoise[i] = hnoise[i - 1] * (0.5 * alpha + (i - 1)) / i
    zero_pad = np.zeros(N)
    hnoise = np.concatenate((hnoise, zero_pad))
    white_noise = np.concatenate((white_noise, zero_pad))
    fh = fft(hnoise)
    fw = fft(white_noise)
    fw = fw * fh
    fw[len(fw) // 2 :] = 0

    fw[0] = fw[0] / 2
    fw[N - 1] = fw[N - 1] / 2

    xi = ifft(fw)
    return 2 * np.real(xi[:N])
