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

`/path/to/inputs` holds three raw little-endian Float64 files, C order, no header, the
numbers in each name being its shape:

```
G_ej_{emitters}x{transceivers}x{samples}_f64le.bin       emitter -> transceiver
G_jr_{receivers}x{transceivers}x{samples}_f64le.bin      transceiver -> receiver
target_h0_len{4 * samples - 3}_f64le.bin                 the design target
```

The example reads the sizes off those names, so a directory of any size works; the paper's own
set is `G_ej_2x28x7680_f64le.bin`, `G_jr_101x28x7680_f64le.bin`, `target_h0_len30717_f64le.bin`.
`python3 test/make_synthetic.py OUTDIR --samples N --receivers n --transceivers n` writes a
stand-in set in the same layout at any size, and `python3 test/verify.py --inputs DIR` runs the
check on any such directory. The routed 0-based `(receiver, emitter)` pairs are `(19, 0)` and
`(79, 1)` on the paper's set and evenly spread otherwise; `default_pairs` in
`examples/run_design.jl` is where to change them.

## Design from a Green's-function file

Reads `G_ej` and `G_jr` from an `.npz`, places the target pulse at a chosen sample, and writes
the filter as a `(transceivers, taps)` `.npy`. Keep `--delay` below about `--M`.

```
julia -t 6 --project=. scripts/design_filter.jl --greens Greens.npz --pairs "19,0;79,1" --c 0.1 --maxiter 100 --M 10000 --delay 6000 --normalize spectrum --out filter.npy
```

The header of `scripts/design_filter.jl` lists every option and its default.

## Score a filter

The isolation metric the paper quotes, here on a generated set so it runs without data:

```
python3 test/make_synthetic.py inputs
julia -t 6 --project=. examples/run_design.jl inputs filter.bin
```

```python
import sys; sys.path.insert(0, "analysis"); import metrics, numpy as np
G_ej = np.fromfile("inputs/G_ej_2x12x512_f64le.bin").reshape(2, 12, 512)     # emitter legs
G_jr = np.fromfile("inputs/G_jr_24x12x512_f64le.bin").reshape(24, 12, 512)   # receiver legs
a = np.fromfile("filter.bin").reshape(12, 1023)           # (transceivers, taps = 2N-1)
print(metrics.isolation(metrics.energies(G_ej, G_jr, a), [(8, 0), (16, 1)]))
```

With your own data, use the shapes in your file names and the pairs the design routed.

## Compare with a direct least-squares solver

Designs the same filter by the normal equations (`design`) and by LSQR on the least-squares
system itself (`design_lsqr`), and prints how far the two filters differ, their isolation and
both run times. Takes an input directory as above, or none for the synthetic set.

```
julia -t 6 --project=. examples/compare_lsqr.jl [/path/to/inputs]
```

## Threads

`julia -t N` sets the solver's thread count; the filter does not depend on it.
`python3 test/verify.py --threads N` passes the same flag through. The Python parts follow
the usual BLAS environment variables, e.g. `OMP_NUM_THREADS=6`.
