#!/usr/bin/env python3
"""Write a tiny synthetic stand-in for the measured inputs, so the repository can be
checked end to end with no data and no network.

    python3 test/make_synthetic.py [outdir] [--samples N] [--emitters n] [--receivers n]
                                   [--transceivers n] [--seed s]

The default ``outdir`` is a directory in the system temporary folder -- the one
``test/verify.py`` uses, so that a check leaves nothing inside the repository.  Everything
is a function of the seed, so two runs produce identical bytes and a directory is reused.

Files written -- raw little-endian float64, C order, no header, named the way the measured
inputs are named, the numbers in each name being its shape, so that one loader
(``RISDesign.load_inputs``, ``test/verify.py``) reads either set:

    G_ej_{n_e}x{K}x{N}_f64le.bin     (n_e, K, N)      G_ej[i, k, n], emitter i -> transceiver k
    G_jr_{n_r}x{K}x{N}_f64le.bin     (n_r, K, N)      G_jr[j, k, n], transceiver k -> receiver j
    target_h0_len{4N-3}_f64le.bin    (4N-3,)          the design target on the Nmin grid

Python reads them with ``np.fromfile(p).reshape(n_e, K, N)``; Julia reads the same bytes
with ``read!(p, Array{Float64}(undef, N, K, n_e))`` and indexes them ``[n, k, i]``.  The
measured set is ``(2, 28, 7680)``, ``(101, 28, 7680)`` and ``30717``; the default here is
1/15 of the sample length and a fraction of the channel count.

What the synthetic legs are.  Each leg is a decaying random impulse response: silence up
to a random arrival sample, then white Gaussian noise under an exponential envelope --
the crudest thing that still has the two properties the design depends on, namely a
broadband response and legs that are independent from one another.  It is not a model of
a room; it only has to exercise the same arithmetic.

The target ``h0`` is built exactly as the measured one is: a ``2*Nf``-tap FIR band-pass
(120 Hz to 12 kHz at 44.1 kHz, Tukey(0.3) window, ``scipy.signal.firwin``), transformed
with ``rfft``, the last bin dropped, and placed on the ``Nmin = 4N-3`` grid that the
solvers expect.  The band is fixed in Hz, so a shorter ``N`` simply resolves it coarsely.

Size of the default problem.  Plain CG stops being reproducible once it loses
orthogonality -- the note ``test/verify.py`` prints says why -- and on a small problem
that happens early.  Run the reference design against itself with nothing changed but the
FFT grid: with 6 transceivers and 8 receivers the two runs already differ by 1e-10 at
iteration 50 and by 1e-6 at iteration 60, far too early for a 60-iteration comparison to
mean anything.  The wall moves out when the problem gets more channels, not when it gets
longer, so the default is ``N=512`` with 2 emitters, 24 receivers and 12 transceivers:
that same grid-to-grid test then stays at 3e-14 through iteration 60 and first reaches
1e-10 near iteration 75, which is where the measured problem's wall sits as well.  The
whole check runs in a couple of seconds.
"""

import argparse
import os
import tempfile

import numpy as np
from scipy.signal import firwin

FS = 44100.0                    # sampling rate of the measurements this imitates
F_LO, F_HI = 120.0, 12000.0     # band of the design target, in Hz
TUKEY_ALPHA = 0.3

DEFAULTS = dict(samples=512, emitters=2, receivers=24, transceivers=12, seed=20240916)
DEFAULT_DIR = os.path.join(tempfile.gettempdir(), "ris_synthetic")   # never inside the repository


def responses(rng, n_path, n_trx, n_samples):
    """``(n_path, n_trx, n_samples)`` decaying random impulse responses.

    Response ``[p, k]`` is zero before its arrival sample ``d`` (uniform over the first
    ``N/16`` samples) and white Gaussian noise times ``exp(-(n-d)/tau)`` after it, with
    ``tau = N/8``, so the tail has died by the end of the window.
    """
    n = np.arange(n_samples)
    d = rng.integers(0, max(n_samples // 16, 1), size=(n_path, n_trx, 1))
    env = np.where(n >= d, np.exp(-(n - d) / (n_samples / 8.0)), 0.0)
    return rng.standard_normal((n_path, n_trx, n_samples)) * env


def target(n_samples):
    """The design target ``h0``, length ``Nmin = 4N-3``.

    ``irfft`` of the first ``Nf = Nmin//2+1`` bins of the spectrum of a ``2*Nf``-tap
    Tukey-windowed FIR band-pass -- the construction the measured target uses.
    """
    nmin = 4 * n_samples - 3
    nf = nmin // 2 + 1
    fir = firwin(2 * nf, [F_LO, F_HI], pass_zero=False,
                 window=("tukey", TUKEY_ALPHA), fs=FS, scale=True)
    return np.fft.irfft(np.fft.rfft(fir)[:nf], nmin)


def make(samples=DEFAULTS["samples"], emitters=DEFAULTS["emitters"],
         receivers=DEFAULTS["receivers"], transceivers=DEFAULTS["transceivers"],
         seed=DEFAULTS["seed"]):
    """``(G_ej, G_jr, h0)`` for one synthetic problem, determined entirely by `seed`."""
    rng = np.random.default_rng(seed)
    return (responses(rng, emitters, transceivers, samples),
            responses(rng, receivers, transceivers, samples),
            target(samples))


def filenames(samples, emitters, receivers, transceivers):
    """The three file names of a problem of this size, in the order ``(G_ej, G_jr, h0)``."""
    return (f"G_ej_{emitters}x{transceivers}x{samples}_f64le.bin",
            f"G_jr_{receivers}x{transceivers}x{samples}_f64le.bin",
            f"target_h0_len{4 * samples - 3}_f64le.bin")


def write(outdir, G_ej, G_jr, h0):
    """Write the three arrays into `outdir` and return their paths."""
    names = filenames(G_ej.shape[2], G_ej.shape[0], G_jr.shape[0], G_ej.shape[1])
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for name, arr in zip(names, (G_ej, G_jr, h0)):
        path = os.path.join(outdir, name)
        np.ascontiguousarray(arr).astype("<f8", copy=False).tofile(path)
        paths.append(path)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("outdir", nargs="?", default=DEFAULT_DIR)
    for name, value in DEFAULTS.items():
        p.add_argument("--" + name, type=int, default=value)
    args = p.parse_args()

    G_ej, G_jr, h0 = make(args.samples, args.emitters, args.receivers,
                          args.transceivers, args.seed)
    for path, arr in zip(write(args.outdir, G_ej, G_jr, h0), (G_ej, G_jr, h0)):
        print(f"wrote {path}  shape {arr.shape}  {os.path.getsize(path) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
