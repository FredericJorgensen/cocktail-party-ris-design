# cocktail-party-ris-design

Code and measured data for *Optimization of Active Reconfigurable Surfaces for Spatiotemporal
MU-MIMO Wave Routing in Disordered Media: Applications to the Cocktail-Party Problem*.

- `RISDesign.jl`: the filter design, by conjugate gradients on the normal equations (`design`)
  and, for comparison, by LSQR (`design_lsqr`).
- `reproduce.jl`: redesigns the optimized filters of Figs. 2 and 3 from the measured Green's
  functions and prints the solver results of the text (design time, LSQR comparison, condition
  numbers).
- `figures.py`: draws Figs. 2 and 3 (`fig2.pdf`, `fig3.pdf`) from the measured recordings and prints
  the numbers the Results quote from them.

## Run

Julia 1.11 and Python 3 with NumPy, SciPy, Matplotlib and Pillow. From the repository root:

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -t 6 --project=. reproduce.jl
python3 figures.py
```

## Data

Measured on 22 September 2026 on the experimental setup of F. Dorlot, C. Bourdeloux, M. Fink and
F. Lemoult, *Reconfigurable and active time-reversal metasurface turns walls into sound routers*,
Commun. Phys. **8**, 446 (2025), https://doi.org/10.1038/s42005-025-02351-3. Sampled at 44.1 kHz. There are 2 emitters, 28 transceivers and 101
receivers at x = -400, -392, ..., 400 mm.

| file | content |
| --- | --- |
| `G_ej.npy` | Green's functions emitter → transceiver, (2, 28, 7680), float64 |
| `G_jr_1.npy`, `G_jr_2.npy` | Green's functions transceiver → receiver, receivers 0–50 and 51–100, (receiver, 28, 7680) |
| `chirp.npy` | the probe chirp, 1 s, 120 Hz to 12 kHz |
| `single.npz`, `double.npz` | Figs. 2 and 3: the routed `pairs`; the filters played on the surface, `tr_filter` (corrected time reversal) and `opt_filter` (optimized), (28, 10000); the recordings `tr_rec` and `opt_rec` at all receivers, (emitter, receiver, sample), as 24-bit PCM counts (full scale 2^23); in `double.npz` also `sup_rec`, the recording of the superposed filters |
| `photo_setup.jpg` | the receiver side of the setup, Fig. 2(e) |
