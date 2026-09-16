"""Evaluation metrics for a designed filter: energies, isolation, crosstalk curves.

The designed filter is *evaluated* by simulating what every emitter delivers to every
receiver.  Emitter ``i`` reaches receiver ``j`` through transceiver ``k`` along the
composite path ``g_ikj = G_ej[i,k] * G_jr[j,k]`` (``*`` = convolution), so replaying the
filter ``a_k`` on every transceiver produces the impulse response

    y_ij(t) = sum_k (g_ikj * a_k)(t),      length conv_len + M - 1 = (2N-1) + M - 1

whose energy ``E[i,j] = sum_t y_ij(t)^2`` is the one number the metrics below use.  The
design puts energy on the routed pairs and suppresses it elsewhere; `isolation` scores
that in dB and `crosstalk_curve` is what the crosstalk plots of the paper show.

Array conventions, matching the rest of the repository:

    G_ej    (n_em, K, N)     emitter legs,  G_ej[i, k] = emitter i -> transceiver k
    G_jr    (n_rec, K, N)    receiver legs, G_jr[j, k] = transceiver k -> receiver j
    filt    (K, M)           one filter per transceiver -- exactly the layout obtained by
                             reading the solver's output with
                             ``np.fromfile(path).reshape(K, M)``
    pairs   [(j, i), ...]    0-based (receiver, emitter) routes, as passed to `design`
    E       (n_em, n_rec)    energies, the output of `energies`

Only NumPy is needed.

    import numpy as np, metrics
    G_ej = np.fromfile("G_ej_2x28x7680_f64le.bin").reshape(2, 28, 7680)
    G_jr = np.fromfile("G_jr_101x28x7680_f64le.bin").reshape(101, 28, 7680)
    a    = np.fromfile("filter.bin").reshape(28, 15359)
    E    = metrics.energies(G_ej, G_jr, a)
    print(metrics.isolation(E, [(19, 0), (79, 1)]))      # dB, one entry per emitter
"""

import numpy as np

__all__ = ["energies", "isolation", "crosstalk_curve"]


def _fast_len(n):
    """Smallest 5-smooth integer >= `n` (a length every FFT implementation likes)."""
    best = 2 ** int(np.ceil(np.log2(max(n, 1))))
    f2 = 1
    while f2 < best:
        f3 = f2
        while f3 < best:
            f5 = f3
            while f5 < n:
                f5 *= 5
            best = min(best, f5)
            f3 *= 3
        f2 *= 2
    return max(best, 1)


def energies(G_ej, G_jr, filt, nfft=None, chunk=16):
    """Energy of the simulated impulse response of every (emitter, receiver) pair.

        E[i, j] = sum_t y_ij(t)^2,     y_ij = sum_k (G_ej[i,k] * G_jr[j,k]) * filt[k]

    Evaluated in the frequency domain, where the convolutions are products and Parseval
    turns the energy into a sum over the rfft bins:

        Y_ij(f) = sum_k Ghat_ej[i,k,f] * Ghat_jr[j,k,f] * Ahat[k,f]
        E[i, j] = (1/nfft) * sum_f wgt[f] * |Y_ij(f)|^2,   wgt = 2 except 1 at DC and,
                                                           for even nfft, at Nyquist

    Both steps are exact, not approximations: the response is a finite sequence of
    ``conv_len + M - 1`` samples (``conv_len = 2N-1``), so any grid at least that long
    holds it without wrap-around, and Parseval's identity is then an equality.  `nfft`
    defaults to the smallest 5-smooth length that is long enough.

    The receivers are processed in blocks of `chunk`, so the full (n_rec, K, nfft/2+1)
    stack of receiver-leg spectra -- 0.7 GB for the measured set -- is never allocated.

    Parameters
    ----------
    G_ej : (n_em, K, N) array           emitter legs
    G_jr : (n_rec, K, N) array          receiver legs
    filt : (K, M) array                 one filter per transceiver
    nfft : int, optional                evaluation grid, must be >= conv_len + M - 1
    chunk : int                         receivers evaluated per block

    Returns
    -------
    E : (n_em, n_rec) array             energies, in the units of the inputs squared
    """
    G_ej = np.asarray(G_ej, dtype=float)
    G_jr = np.asarray(G_jr, dtype=float)
    a = np.asarray(filt, dtype=float)
    if G_ej.ndim != 3 or G_jr.ndim != 3 or a.ndim != 2:
        raise ValueError("expected G_ej (n_em, K, N), G_jr (n_rec, K, N), filt (K, M)")
    if G_ej.shape[1:] != G_jr.shape[1:]:
        raise ValueError(f"leg shapes disagree: {G_ej.shape} vs {G_jr.shape}")
    if a.shape[0] != G_ej.shape[1]:
        raise ValueError(f"filt has {a.shape[0]} rows, but there are {G_ej.shape[1]} "
                         "transceivers")

    n_em, _, N = G_ej.shape
    n_rec, M = G_jr.shape[0], a.shape[1]
    nmin = (2 * N - 1) + M - 1                  # exact length of the simulated response
    nfft = _fast_len(nmin) if nfft is None else int(nfft)
    if nfft < nmin:
        raise ValueError(f"nfft must be >= conv_len + M - 1 = {nmin}, got {nfft}")

    # Emitter leg and filter are shared by all receivers, so fold them together once.
    B = np.fft.rfft(G_ej, nfft, axis=-1) * np.fft.rfft(a, nfft, axis=-1)    # (n_em, K, nf)

    wgt = np.full(B.shape[-1], 2.0)             # the bins rfft does not store, counted twice
    wgt[0] = 1.0
    if nfft % 2 == 0:
        wgt[-1] = 1.0

    E = np.empty((n_em, n_rec))
    for lo in range(0, n_rec, max(int(chunk), 1)):
        block = slice(lo, min(lo + max(int(chunk), 1), n_rec))
        Gj = np.fft.rfft(G_jr[block], nfft, axis=-1)                        # (nb, K, nf)
        Y = np.einsum("jkf,ikf->ijf", Gj, B, optimize=True)                 # (n_em, nb, nf)
        E[:, block] = (wgt * (Y.real ** 2 + Y.imag ** 2)).sum(-1) / nfft
    return E


def isolation(E, pairs, excl=2):
    """Isolation in dB, one entry per emitter -- the metric quoted in the paper.

    For emitter ``i`` with routed target receivers ``T_i = {j : (j, i) in pairs}``:

        isolation[i] = min over t in T_i of  10 * log10( E[i,t] / mean(E[i, off_i]) )

    where the off-target set ``off_i`` is every receiver that is more than `excl`
    positions away from *all* of that emitter's targets -- the guard band keeps the
    immediate neighbours of a target, which no finite aperture can separate from it,
    out of the leakage average.  Taking the minimum over the targets makes a
    multi-target emitter scored by its worst-served target.

    Parameters
    ----------
    E : (n_em, n_rec) array     energies from `energies`
    pairs : list of (j, i)      0-based (receiver, emitter) routes
    excl : int                  guard half-width, in receiver positions

    Returns
    -------
    (n_em,) array of dB values
    """
    E = np.asarray(E, dtype=float)
    if E.ndim != 2:
        raise ValueError("E must be (n_em, n_rec)")
    n_em, n_rec = E.shape

    out = np.empty(n_em)
    for i in range(n_em):
        targets = [j for (j, e) in pairs if e == i]
        if not targets:
            raise ValueError(f"emitter {i} has no routed pair in {pairs}")
        offtarget = np.ones(n_rec, dtype=bool)
        for t in targets:
            if not 0 <= t < n_rec:
                raise ValueError(f"receiver {t} out of range for E with {n_rec} receivers")
            offtarget[max(0, t - excl):t + excl + 1] = False   # max(): no negative-index wrap
        if not offtarget.any():
            raise ValueError(f"emitter {i}: the guard bands cover every receiver")
        leak = E[i, offtarget].mean()
        out[i] = min(10 * np.log10(E[i, t] / leak) for t in targets)
    return out


def crosstalk_curve(E, pairs=None):
    """Per-emitter energy profile over the receivers, in dB relative to the target.

        crosstalk[i, j] = 10 * log10( E[i, j] / ref_i )

    with ``ref_i = max_j E[i, j]`` by default, which is the reference the crosstalk
    plots use: a working design puts that maximum on the routed target, so the curve
    peaks at 0 dB there and every other value is the leakage into that receiver.  Pass
    `pairs` to normalise by the routed target itself (``ref_i = max energy over that
    emitter's targets``) instead, which is the honest reference when a design fails and
    the loudest receiver is not the intended one.

    Parameters
    ----------
    E : (n_em, n_rec) array     energies from `energies`
    pairs : list of (j, i), optional    0-based (receiver, emitter) routes

    Returns
    -------
    (n_em, n_rec) array of dB values, <= 0 everywhere when `pairs` is omitted
    """
    E = np.asarray(E, dtype=float)
    if E.ndim != 2:
        raise ValueError("E must be (n_em, n_rec)")
    if pairs is None:
        ref = E.max(axis=1, keepdims=True)
    else:
        ref = np.empty((E.shape[0], 1))
        for i in range(E.shape[0]):
            targets = [j for (j, e) in pairs if e == i]
            if not targets:
                raise ValueError(f"emitter {i} has no routed pair in {pairs}")
            ref[i, 0] = max(E[i, t] for t in targets)
    return 10 * np.log10(E / ref)
