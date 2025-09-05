import numpy as np
from scipy import signal, interpolate


def square_demod(x, b, *options):
    """
    Demodulate a square-wave modulated signal.

    Parameters
    ----------
    x : ndarray (N, M)
        Modulated signal. Each column is a channel.
    b : ndarray (N,)
        Chop reference signal (0 or 1).
    *options : str
        Optional flags:
            'deglitch'       - filter glitches in chop ref
            'expand'         - expand C/S to same size as x using 'nearest'
            'linear', 'spline', 'nearest' - interpolation method
            'keep_single'    - keep singleton high/low regions
            'highres'        - treat `b` as high-resolution chop reference

    Returns
    -------
    c : ndarray
        Cosine demodulated component.
    s : ndarray
        Sine demodulated component.
    ic : ndarray
        Indices corresponding to cosine cycles.
    is_ : ndarray
        Indices corresponding to sine cycles.
    """
    DO_DEGLITCH = False
    DO_INTERP = ''
    NAN_SINGLE = True
    HIGH_RES = False

    for arg in options:
        arg = arg.lower()
        if arg == 'deglitch':
            DO_DEGLITCH = True
        elif arg == 'expand':
            DO_INTERP = 'nearest'
        elif arg in ['linear', 'spline', 'nearest']:
            DO_INTERP = arg
        elif arg == 'keep_single':
            NAN_SINGLE = False
        elif arg == 'highres':
            HIGH_RES = True
        else:
            raise ValueError(f"Unknown option: {arg}")

    b = np.asarray(b).flatten()
    if np.any((b > 1) | (b < 0) | ~np.isfinite(b)):
        raise ValueError("Chop reference b should be in range 0–1 and finite.")
    b = (b >= 0.5)

    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = x.T

    N = x.shape[0]
    ind_du, ind_ud = np.array([]), np.array([])

    if HIGH_RES:
        b = b.astype(float)
        i_hires = np.linspace(1, N, len(b))
        dif_hires = np.diff(np.insert(b, 0, b[0]))
        ind_du = i_hires[dif_hires > 0]
        ind_ud = i_hires[dif_hires < 0]
        b = interpolate.interp1d(i_hires, b, kind='linear', fill_value='extrapolate')(np.arange(1, N + 1))
        b = (b > 0.5)

    if DO_DEGLITCH:
        t = np.arange(len(b))
        b_scaled = (b - 0.5) * 2
        f, p = signal.welch(b_scaled, fs=1)
        p[:4], p[-4:] = 0, 0
        fchop = f[np.argmax(p)]
        chcos = np.cos(2 * np.pi * fchop * t)
        chsin = np.sin(2 * np.pi * fchop * t)
        bflt, aflt = signal.butter(2, 2 * fchop / 5, btype='low')
        bcos = signal.filtfilt(bflt, aflt, chcos * b_scaled)
        bsin = signal.filtfilt(bflt, aflt, chsin * b_scaled)
        ph = np.arctan2(bsin, bcos)
        b = (np.cos(2 * np.pi * fchop * t - ph) > 0).astype(np.float32)

    j = {
        2: np.where(np.concatenate([b, [False]]) & np.concatenate([[True], ~b]))[0]+1,
        4: np.where(np.concatenate([~b, [False]]) & np.concatenate([[False], b]))[0]+1 
    }

    if not b[-1]:
        nlow = len(b) - j[4][-1] + 1
        nave = np.mean(np.diff(j[2]))
        if nlow >= nave / 4 and nlow >= 2:
            j[2] = np.append(j[2], len(b) + 1)
            if HIGH_RES:
                ind_du = np.append(ind_du, len(b) + 1)

    if len(j[2]) < 2 or len(j[4]) < 2:
        raise ValueError("Need at least two chop cycles.")

    if HIGH_RES:
        if b[0]:
            ind_du = np.insert(ind_du, 0, 1)
        tmp1 = np.vstack((j[2][j[2] < j[4][-1]], j[4][j[4] > j[2][0]])).T
        tmp2 = np.vstack((ind_du[ind_du < ind_ud[-1]], ind_ud[ind_ud > ind_du[0]])).T
        j[3] = np.round(np.mean(tmp1, axis=1) + 1e-5 * (np.mean(tmp2, axis=1) - np.mean(tmp1, axis=1)))
        tmp1 = np.vstack((j[4][j[4] < j[2][-1]], j[2][j[2] > j[4][0]])).T
        tmp2 = np.vstack((ind_ud[ind_ud < ind_du[-1]], ind_du[ind_du > ind_ud[0]])).T
        j[1] = np.round(np.mean(tmp1, axis=1) + 1e-5 * (np.mean(tmp2, axis=1) - np.mean(tmp1, axis=1)))
    else:
        # j[3]: midpoint of rising (j[2]) and next falling (j[4])
        j2_lhs = j[2][j[2] < j[4][-1]]
        j4_rhs = j[4][j[4] > j[2][0]]
        n_mid = min(len(j2_lhs), len(j4_rhs))
        j[3] = np.round((j2_lhs[:n_mid] + j4_rhs[:n_mid]) / 2)

        # j[1]: midpoint of falling (j[4]) and next rising (j[2])
        j4_lhs = j[4][j[4] < j[2][-1]]
        j2_rhs = j[2][j[2] > j[4][0]]
        n_mid = min(len(j4_lhs), len(j2_rhs))
        j[1] = np.round((j4_lhs[:n_mid] + j2_rhs[:n_mid]) / 2)

        #tmp = np.column_stack([j[2][j[2] < j[4][-1]],j[4][j[4] > j[2][0]]])
        #j[3] = np.round(np.mean(tmp, axis=1) + 1e-5 * (np.random.rand(len(tmp)) - 0.5)).astype(int)
        #tmp = np.column_stack([j[4][j[4] < j[2][-1]],j[2][j[2] > j[4][0]]])
        #j[1] = np.round(np.mean(tmp, axis=1) + 1e-5 * (np.random.rand(len(tmp)) - 0.5)).astype(int)


    if not b[0]:
        j[1] = np.insert(j[1], 0, 1)
    if not b[-1] and j[2][-1] != len(b):
        j[1] = np.append(j[1], len(b)) 

    if len(j[1]) > len(j[2]):
        j[1] = j[1][:len(j[2])]


    c, ic = demod1(j, [1, 2, 3, 4], x, NAN_SINGLE)
    s, is_ = demod1(j, [2, 3, 4, 1], x, NAN_SINGLE)

    if DO_INTERP:
        c_full = np.zeros_like(x)
        s_full = np.zeros_like(x)
        for i in range(x.shape[1]):
            interp_c = interpolate.interp1d(ic, c[:, i], kind=DO_INTERP, fill_value='extrapolate', bounds_error=False)
            interp_s = interpolate.interp1d(is_, s[:, i], kind=DO_INTERP, fill_value='extrapolate', bounds_error=False)
            jitter = 1e-5 * (np.random.rand(N) - 0.5)
            jj = np.arange(N) + jitter
            c_full[:, i] = interp_c(jj)
            s_full[:, i] = interp_s(jj)
        c = c_full
        s = s_full

    return c, s, ic, is_


def demod1(j, ph, x, nan_singletons):
    """
    Demodulate using explicit cycle-based integration to match MATLAB semantics.
    j : dict of {1,2,3,4} phase transition indices
    ph : phase order for demodulation (e.g. [1,2,3,4] or [2,3,4,1])
    x : input signal (N, M)
    nan_singletons : if True, output NaN for singleton phases
    """
    sgn = np.array([-1, 1, 1, -1])
    jj = np.zeros((len(j[ph[0]]) - 1, 5), dtype=int)

    # Align j indices
    for i in range(4):
        valid = (j[ph[i]] >= j[ph[0]][0]) & (j[ph[i]] <= j[ph[0]][-1])
        tmpj = j[ph[i]][valid].astype(int)
        jj[:, i] = tmpj[:jj.shape[0]]
    jj[:, 4] = j[ph[0]][j[ph[0]] > j[ph[0]][0]].astype(int)[:jj.shape[0]]

    Ncycles = jj.shape[0]
    Nsamples, Nch = x.shape
    z = np.zeros((Ncycles, Nch))
    ising = np.any(np.diff(jj[:, :4], axis=1) == 1, axis=1)  # singleton detector

    for k in range(Ncycles):
        val = np.zeros(Nch)
        for i in range(4):
            js = jj[k, i] - 1
            je = jj[k, i + 1] - 1
            js = max(js, 0)
            je = min(je, Nsamples)

            if js >= je:
                if nan_singletons:
                    val[:] = np.nan
                    break
                else:
                    continue

            seg = x[js:je, :]
            mean_seg = np.mean(seg, axis=0) if seg.shape[0] > 0 else np.zeros(Nch)
            val += sgn[i] * mean_seg

        if nan_singletons and ising[k]:
            z[k, :] = np.nan
        else:
            z[k, :] = val / 2  # as in MATLAB
    iz = jj[:, 2]  # middle index
    return z, iz