import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import warnings


def square_demod(x, b, *args):
    # Parse options
    do_deglitch = False
    do_interp = ''
    nan_single = True
    high_res = False

    for arg in args:
        if arg.lower() == 'deglitch':
            do_deglitch = True
        elif arg.lower() == 'expand':
            do_interp = 'nearest'
        elif arg.lower() in ['linear', 'spline', 'nearest']:
            do_interp = arg.lower()
        elif arg.lower() == 'keep_single':
            nan_single = False
        elif arg.lower() == 'highres':
            high_res = True
        else:
            raise ValueError(f"Unknown option {arg}")

    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = x.T

    b = np.asarray(b).flatten()
    if np.any(~np.isfinite(b)) or np.max(b) > 1 or np.min(b) < 0:
        raise ValueError("Chop reference b should be in range 0-1.")

    b = (b >= 0.5).astype(np.float32)

    if high_res:
        i_hires = np.linspace(0, len(x) - 1, len(b))
        dif_hires = np.diff(np.concatenate([[b[0]], b]))
        ind_du = i_hires[dif_hires > 0]
        ind_ud = i_hires[dif_hires < 0]
        b = np.interp(np.arange(len(x)), i_hires, b) > 0.5

    if do_deglitch:
        bflt, aflt = butter(2, 0.4)
        t = np.arange(len(b))
        fchop = find_chop_freq(b)  # custom PSD logic can be added
        chcos = np.cos(2 * np.pi * fchop * t)
        chsin = np.sin(2 * np.pi * fchop * t)
        bcos = filtfilt(bflt, aflt, chcos * b)
        bsin = filtfilt(bflt, aflt, chsin * b)
        ph = np.arctan2(bsin, bcos)
        b = np.cos(2 * np.pi * fchop * t - ph) > 0

    j = {}
    j[2] = np.where(np.concatenate([[False], b[:-1] < 0.5]) & (b > 0.5))[0]
    j[4] = np.where(np.concatenate([[False], b[:-1] > 0.5]) & (b < 0.5))[0]

    if not b[-1] and len(j[4]) > 0:
        nlow = len(b) - j[4][-1]
        nave = np.mean(np.diff(j[2])) if len(j[2]) > 1 else 0
        if nlow >= nave / 4 and nlow >= 2:
            j[2] = np.append(j[2], len(b))

    if len(j[2]) < 2 or len(j[4]) < 2:
        raise ValueError("Need at least two chop cycles.")

    # Get 90-degree offset indices
    if high_res:
        if b[0]:
            ind_du = np.insert(ind_du, 0, 0)
        j[3] = np.round(0.5 * (j[2][:-1] + j[4][1:])).astype(int)
        j[1] = np.round(0.5 * (j[4][:-1] + j[2][1:])).astype(int)
    else:
        min_len = min(len(j[2]) - 1, len(j[4]) - 1)
        j2_cut = j[2][:min_len + 1]
        j4_cut = j[4][:min_len + 1]

        j[3] = np.round(0.5 * (j2_cut[:-1] + j4_cut[1:]) + 1e-5 * (np.random.rand(min_len) - 0.5)).astype(int)
        j[1] = np.round(0.5 * (j4_cut[:-1] + j2_cut[1:]) + 1e-5 * (np.random.rand(min_len) - 0.5)).astype(int)


    if not b[0]:
        j[1] = np.insert(j[1], 0, 0)
    if not b[-1] and (j[2][-1] != len(b)):
        j[1] = np.append(j[1], len(b))

    if len(j[1]) < 2 or len(j[3]) < 2:
        raise ValueError("Need at least two chop cycles.")

    c, ic = demod1(j, [1, 2, 3, 4], x, nan_single)
    s, is_ = demod1(j, [2, 3, 4, 1], x, nan_single)

    if do_interp:
        ctmp = np.zeros_like(x)
        stmp = np.zeros_like(x)
        jj = np.arange(x.shape[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for i in range(x.shape[1]):
                fc = interp1d(ic, c[:, i], kind=do_interp, bounds_error=False, fill_value='extrap')
                fs = interp1d(is_, s[:, i], kind=do_interp, bounds_error=False, fill_value='extrap')
                ctmp[:, i] = fc(jj)
                stmp[:, i] = fs(jj)
        c = ctmp
        s = stmp

    return c, s, ic, is_


def demod1(j, ph, x, nan_singletons):
    jj = np.zeros((min(len(j[ph[0]]) - 1, len(j[ph[1]]), len(j[ph[2]]), len(j[ph[3]])), 5), dtype=int)
    for i in range(4):
        jj[:, i] = j[ph[i]][:jj.shape[0]]
    jj[:, 4] = j[ph[0]][1:jj.shape[0] + 1]
    sgn = np.array([-1, 1, 1, -1])
    mul = np.zeros(x.shape[0])
    for i in range(4):
        js = jj[:, i]
        je = jj[:, i + 1]
        n = np.zeros(x.shape[0] + 1)
        np.add.at(n, js, 1. / np.maximum(1, je - js))
        np.add.at(n, je, -1. / np.maximum(1, je - js))
        mul += sgn[i] * np.cumsum(n[:-1])

    z = np.zeros((jj.shape[0], x.shape[1]))
    for k in range(x.shape[1]):
        tmpx = np.nan_to_num(x[:, k])
        tmpx = np.cumsum(tmpx * mul)
        z[:, k] = np.diff(np.concatenate([[0], tmpx[jj[:, 4] - 1]]))

    ising = np.any(np.diff(jj[:, :4], axis=1) == 1, axis=1)
    if nan_singletons:
        z[ising, :] = np.nan
    else:
        for i in range(4):
            js = jj[ising, i]
            je = jj[ising, i + 1]
            if i in [0, 2]:
                js[je == js] = js[je == js] - 1
            else:
                je[je == js] = je[je == js] + 1
            for k in range(z.shape[1]):
                for idx, (s_, e_) in enumerate(zip(js, je)):
                    z[ising[idx], k] += sgn[i] * np.mean(x[s_:e_, k])
    z /= 2
    iz = jj[:, 2]
    return z, iz