from math import sqrt

import numpy as np

from .config import GAMMA
from .fields import h_eff


def pure_llg(alpha: float, gamma: float, m: np.ndarray, h, dt: float):
    mxh = np.cross(m, h)
    pref = gamma / (1 + alpha**2)
    dmdt = -pref * mxh - alpha * pref * np.cross(m, mxh)
    m += dt * dmdt
    return m / np.linalg.norm(m, axis=3, keepdims=True)


def ll_operator(alpha: float, gamma: float, m: np.ndarray, h):
    mxh = np.cross(m, h)
    pref = gamma / (1 + alpha**2)
    return -pref * mxh - alpha * pref * np.cross(m, mxh)


def euler_heun(
    alpha: float,
    gamma: float,
    m: np.ndarray,
    h: np.ndarray,
    h_noise: np.ndarray,
    dt: float,
):
    fn = m + dt * ll_operator(alpha=alpha, gamma=gamma, m=m, h=h)
    gn = ll_operator(alpha=alpha, gamma=gamma, m=m, h=h_noise)
    m_next = m + gn * sqrt(dt)
    gnapprox = ll_operator(alpha=alpha, gamma=gamma, m=m_next, h=h_noise)
    m = m + fn * dt + 0.5 * (gn + gnapprox) * sqrt(dt)
    return m / np.linalg.norm(m, axis=3, keepdims=True)


def pure_llg_torque(
    alpha: float,
    gamma: float,
    m: np.ndarray,
    h,
    Hdl: float,
    Hfl: float,
    p: list[float],
    dt: float,
):
    mxh = np.cross(m, h)
    mxp = np.cross(m, p)
    pref = gamma / (1 + alpha**2)
    dmdt = -pref * mxh - alpha * pref * np.cross(m, mxh)
    dtaudt = -pref * Hfl * mxp - alpha * pref * Hdl * np.cross(m, mxp)
    m += dt * (dmdt + dtaudt)
    return m / np.linalg.norm(m, axis=3, keepdims=True)


def llg(A, alpha, K, kaxis, ms, m, dt, m_pad, f_n_demag, n, dx, h_zee=0.0):
    m_pad[: n[0], : n[1], : n[2], :] = m
    h = (
        h_eff(
            A=A,
            ms=ms,
            m=m,
            m_pad=m_pad,
            f_n_demag=f_n_demag,
            n=n,
            dx=dx,
            K=K,
            kaxis=kaxis,
        )
        + h_zee
    )
    return pure_llg(alpha=alpha, gamma=GAMMA, m=m.copy(), dt=dt, h=h)


def llg_torque(
    A,
    alpha,
    K,
    kaxis,
    ms,
    m,
    dt,
    m_pad,
    f_n_demag,
    n,
    dx,
    h_zee=0.0,
    Hfl=0.0,
    Hdl=0.0,
    p=(0, 0, 0),
):
    m_pad[: n[0], : n[1], : n[2], :] = m
    h = (
        h_eff(
            A=A,
            ms=ms,
            K=K,
            kaxis=kaxis,
            m=m,
            m_pad=m_pad,
            f_n_demag=f_n_demag,
            n=n,
            dx=dx,
        )
        + h_zee
    )
    return pure_llg_torque(
        alpha=alpha, gamma=GAMMA, m=m.copy(), dt=dt, h=h, Hfl=Hfl, Hdl=Hdl, p=p
    )
