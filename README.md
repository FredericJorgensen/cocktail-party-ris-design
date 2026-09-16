# cocktail-party-ris-design

Filter design for active reconfigurable surfaces that route sound from several emitters to
their own receivers through a disordered medium. Companion code for *Optimization of Active
Reconfigurable Surfaces for Spatiotemporal MU-MIMO Wave Routing in Disordered Media:
Applications to the Cocktail-Party Problem*.

## Install

Julia 1.9 or newer, and Python 3 with NumPy and SciPy.

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
python3 -m pip install numpy scipy
```

## Check that it works

Runs the Julia solver and the NumPy reference implementation on synthetic inputs it
generates itself, and compares them. No data, no network, a few seconds. Exit status 0 when
every criterion passes.

```
python3 test/verify.py
```

## Design a filter on synthetic data

```
julia -t 6 --project=. examples/run_design.jl
```

Writes `filter.bin`: the designed filter, raw Float64, one column of taps per transceiver.

## Design a filter on your own data

```
julia -t 6 --project=. examples/run_design.jl /path/to/inputs filter.bin
```

`/path/to/inputs` holds three raw little-endian Float64 files, C order, no header:

```
G_ej_2x28x7680_f64le.bin       (emitters, transceivers, samples)      emitter -> transceiver
G_jr_101x28x7680_f64le.bin     (receivers, transceivers, samples)     transceiver -> receiver
target_h0_len30717_f64le.bin   (4 * samples - 3,)                     the design target
```

Those are the names the example reads, and `load_inputs(dir; N, K)` takes other sample and
transceiver counts. `python3 test/make_synthetic.py OUTDIR --samples N --receivers n
--transceivers n` writes a set in the same layout at any size, and `python3 test/verify.py
--inputs DIR` runs the check on any such directory. The routed 0-based
`(receiver, emitter)` pairs are set in `examples/run_design.jl`.

Score the result with the paper's isolation metric:

```python
import sys; sys.path.insert(0, "analysis"); import metrics, numpy as np
G_ej = np.fromfile("inputs/G_ej_2x28x7680_f64le.bin").reshape(2, 28, 7680)
G_jr = np.fromfile("inputs/G_jr_101x28x7680_f64le.bin").reshape(101, 28, 7680)
a = np.fromfile("filter.bin").reshape(28, 15359)          # (transceivers, taps)
print(metrics.isolation(metrics.energies(G_ej, G_jr, a), [(19, 0), (79, 1)]))
```

## Threads

`julia -t N` sets the solver's thread count; the filter does not depend on it.
`python3 test/verify.py --threads N` passes the same flag through. The Python parts follow
the usual BLAS environment variables, e.g. `OMP_NUM_THREADS=6`.
