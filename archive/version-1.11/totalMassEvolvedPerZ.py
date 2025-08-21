import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import h5py as h5
import functools

@functools.lru_cache()
def __get_imf_normalisation_values(m1=0.01, m2=0.08, m3=0.5, m4=200.0, a12=0.3, a23=1.3, a34=2.3):
    b1 = 1 / (
            (m2 ** (1 - a12) - m1 ** (1 - a12)) / (1 - a12) +
            m2 ** (a23 - a12) * (m3 ** (1 - a23) - m2 ** (1 - a23)) / (1 - a23) +
            m2 ** (a23 - a12) * m3 ** (a34 - a23) * (m4 ** (1 - a34) - m3 ** (1 - a34)) / (1 - a34)
    )
    b2 = b1 * m2 ** (a23 - a12)
    b3 = b2 * m3 ** (a34 - a23)
    return b1, b2, b3


def __piecewise_kroupa_imf(m, b1=1.0, b2=1.0, b3=1.0, m1=0.01, m2=0.08, m3=0.5, m4=200.0, a12=0.3,
                           a23=1.3, a34=2.3):
    if m1 <= m < m2:
        return b1 * m ** (-a12 - 1)
    if m2 <= m < m3:
        return b2 * m ** (-a23 - 1)
    if m3 <= m <= m4:
        return b3 * m ** (-a34 - 1)
    return 0.0


def analytical_star_forming_mass_per_binary_using_kroupa_imf(m1, m2, m3, m4):
    b1, b2, b3 = __get_imf_normalisation_values()
    def f(m):
        return m * __piecewise_kroupa_imf(m, b1=b1, b2=b2, b3=b3)
    I = quad(f, m1, m4)[0]
    return I


def draw_samples_from_kroupa_imf(n_samples: int, m1=0.01, m2=0.08, m3=0.5, m4=200.0, a12=0.3, a23=1.3, a34=2.3):
    b1, b2, b3 = __get_imf_normalisation_values(m1, m2, m3, m4, a12, a23, a34)
    # Determine how many samples are in each range
    # Probability ratios
    p1 = quad(__piecewise_kroupa_imf, m1, m2, args=(b1, b2, b3, m1, m2, m3, m4, a12, a23, a34))[0]
    p2 = quad(__piecewise_kroupa_imf, m2, m3, args=(b1, b2, b3, m1, m2, m3, m4, a12, a23, a34))[0]
    p3 = quad(__piecewise_kroupa_imf, m3, m4, args=(b1, b2, b3, m1, m2, m3, m4, a12, a23, a34))[0]
    nsamples_in_m1 = int(n_samples * p1 / (p1 + p2 + p3))
    nsamples_in_m2 = int(n_samples * p2 / (p1 + p2 + p3))
    nsamples_in_m3 = n_samples - nsamples_in_m1 - nsamples_in_m2
    # Sample each region
    u1 = np.random.rand(nsamples_in_m1)
    u2 = np.random.rand(nsamples_in_m2)
    u3 = np.random.rand(nsamples_in_m3)
    s1 = m1 * ((1 - u1) + u1 * (m2 / m1) ** (1 - a12)) ** (1 / (1 - a12))
    s2 = m2 * ((1 - u2) + u2 * (m3 / m2) ** (1 - a23)) ** (1 / (1 - a23))
    s3 = m3 * ((1 - u3) + u3 * (m4 / m3) ** (1 - a34)) ** (1 / (1 - a34))
    return np.concatenate([s1, s2, s3])