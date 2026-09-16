"""Reference NumPy implementation of the active-surface design.

This is the slow, literal transcription of the equations the paper solves, kept so
that the fast Julia solver in ``src/RISDesign.jl`` can be checked against something a
reader can verify line by line.  Nothing here is tuned: the composites are formed as
written, the operator is stored as a full ``K x K`` matrix per frequency, and the
conjugate gradients are the textbook three-term recursion.

Model.  Transceiver ``k`` replays the filter ``a_k`` (``M`` taps).  Emitter ``i``
reaches receiver ``j`` through the composite response

    g_ikj = G_ej[i, k] * G_jr[j, k]          (* = convolution, length conv_len = 2N-1)

so the impulse response emitter ``i`` produces at receiver ``j`` is ``sum_k g_ikj * a_k``.
With ``w_ij = 1`` on the routed ``(receiver, emitter)`` pairs and ``w_ij = c`` on every
other combination, and the target ``h_ref_ij = h0`` on the routed pairs and ``0``
elsewhere, the design minimises

    sum_ij w_ij || sum_k g_ikj * a_k - h_ref_ij ||^2      over the M = conv_len taps of a_k

The minimiser solves the normal equations ``U a = b``.  Writing ``ghat_ikj`` for the
spectrum of ``g_ikj`` on an ``Nfft``-point grid, the two pieces are

    R_ks(f) = sum_ij w_ij conj(ghat_ikj(f)) ghat_isj(f)   Hermitian positive semidefinite in (k, s)
    (U a)_k = first M samples of irfft(sum_s R_ks rfft(a_s))
    b[k, n] = sum over the routed pairs of sum_m g_ikj(m) h0(n + m),   n = 0 .. M-1

``b`` is the correlation of the routed composites with the target, evaluated as
``irfft(sum over routed pairs of conj(ghat_ikj) rfft(h0))[:M]``.  Both formulas are
linear, not circular, convolutions because the grid ``Nfft = conv_len + M - 1`` is long
enough to hold every lag that occurs: ``n + m <= (M-1) + (conv_len-1) < Nfft``.  Any
longer grid gives the same operator -- the Julia solver rounds this one up to the
nearest 5-smooth length.  Plain conjugate gradients from ``a = 0`` then run for exactly
``maxiter`` iterations: there is no stopping test, the iteration count is the
regularisation.

Array layout.  Python keeps the sample index last and returns ``a`` with shape
``(K, M) = (transceiver, tap)``; Julia keeps it first and returns ``(M, K) = (tap,
transceiver)``.  The two are transposes of each other: ``design(...) == julia_a.T``
entry for entry, and ``a.tofile(...)`` here writes the same bytes as ``write(io, a)``
there.  The raw inputs follow the same rule: the C-order files ``(2, 28, 7680)`` and
``(101, 28, 7680)`` read as ``G_ej[i, k, n]`` and ``G_jr[j, k, n]`` in Python and as
``[n, k, i]`` and ``[n, k, j]`` in Julia.

Comparing runs.  Plain CG loses orthogonality once the Ritz values start to converge --
near iteration 75 at the paper's size, earlier on small problems.  After that the
iterate is chaotic with respect to round-off, and two faithful implementations, or the
same one on two FFT grids, drift apart in relative L2 while still describing the same
physics.  Judge agreement at 20 and 60 iterations, and at 100 by the physical metrics.

Cost.  Every one of the ``n_e * n_r`` pairs is visited and the full ``K x K`` operator is
kept for each of the ``Nfft/2+1`` frequencies -- 193 MB of operator and about 1.6 GB peak
at the paper's size, where the Julia solver stores one triangle and never forms the
composites.  Use ``src/RISDesign.jl`` for real work.
"""

import numpy as np


def design(G_ej, G_jr, h0, pairs, c, maxiter=100):
    """Design the filters: ``(K, M)`` array, row ``k`` being the filter of transceiver ``k``.

    ``G_ej`` is ``(n_e, K, N)``, ``G_jr`` is ``(n_r, K, N)``, ``h0`` is the length
    ``conv_len + M - 1`` design target, ``pairs`` are the 0-based ``(receiver, emitter)``
    routes, and ``c`` is the weight of every unrouted (emitter, receiver) combination.
    """
    n_e, K, N = G_ej.shape
    n_r = G_jr.shape[0]
    M = conv_len = 2 * N - 1                      # filter length = composite length
    nfft = conv_len + M - 1                       # long enough: no wrap-around anywhere
    if h0.shape != (nfft,):
        raise ValueError(f"h0 must have length {nfft}, got {h0.shape}")
    if len(set(map(tuple, pairs))) != len(pairs):
        raise ValueError("pairs must be unique, a repeated route would be weighted twice")

    w = np.full((n_e, n_r), float(c))             # w_ij = c off the routes, 1 on them
    for j, i in pairs:
        w[i, j] = 1.0

    Ge = np.fft.rfft(G_ej, nfft, axis=-1)         # (n_e, K, nf) zero-padded leg spectra
    Gr = np.fft.rfft(G_jr, nfft, axis=-1)         # (n_r, K, nf)
    nf = Ge.shape[-1]

    R = np.zeros((nf, K, K), dtype=complex)       # R_ks(f), one full matrix per frequency
    for i in range(n_e):
        for j in range(n_r):
            g = (Ge[i] * Gr[j]).T                 # ghat_ikj(f), as (nf, K)
            R += w[i, j] * g.conj()[:, :, None] * g[:, None, :]

    B = np.zeros((K, nf), dtype=complex)          # sum over routes of conj(ghat) * rfft(h0)
    H0 = np.fft.rfft(h0, nfft)
    for j, i in pairs:
        B += (Ge[i] * Gr[j]).conj() * H0
    b = np.fft.irfft(B, nfft, axis=-1)[:, :M]

    def apply_U(x):
        """``(U x)_k``, the first M samples of ``irfft(sum_s R_ks rfft(x_s))``."""
        y = np.einsum("fks,sf->kf", R, np.fft.rfft(x, nfft, axis=-1))
        return np.fft.irfft(y, nfft, axis=-1)[:, :M]

    a = np.zeros((K, M))                          # plain CG on U a = b, from a = 0
    r = b.copy()
    p = r.copy()
    rs = float(np.vdot(r, r).real)
    for _ in range(maxiter):
        Up = apply_U(p)
        alpha = rs / float(np.vdot(p, Up).real)
        a += alpha * p
        r -= alpha * Up
        rs_new = float(np.vdot(r, r).real)
        p = r + (rs_new / rs) * p
        rs = rs_new
    return a
