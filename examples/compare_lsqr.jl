# The design by the normal equations (`design`) against the same objective solved directly by
# LSQR (`design_lsqr`), on the same inputs, at the same iteration counts.
#
#     julia -t 6 examples/compare_lsqr.jl [input_dir]
#
# Inputs as for `run_design.jl`: a directory in the `load_inputs` layout, or none for the
# synthetic set.  Prints, for 20, 60 and 100 iterations, the relative L2 difference of the two
# filters and their isolation (dB per emitter, the paper's metric), then the wall time of each
# solver at 100 iterations, best of five runs.  Exits with status 1 unless the filters agree to
# 1e-10 at 20 iterations and their isolation to 0.01 dB at every count.  Later iterates are not
# gated in L2: once CG loses orthogonality -- near iteration 75 on the measured set, earlier on
# the small synthetic one -- both recursions are chaotic with respect to round-off, as
# `test/verify.py` explains, so there the filters are compared through the physics.

include(joinpath(@__DIR__, "..", "src", "RISDesign.jl"))
using .RISDesign, LinearAlgebra, Printf
include(joinpath(@__DIR__, "inputs.jl"))

const C = 0.1
const ITERS = (20, 60, 100)

"""`E[i, j]`: energy of the response of emitter `i` at receiver `j` to the filter `a`, i.e.
`sum_t (sum_k g_ikj * a_k)(t)^2`, from the unweighted rectangular operator by Parseval."""
function energies(G_ej, G_jr, h0, a)
    N = size(G_ej, 1)
    _, M, nfft = RISDesign.problem_size(G_ej, G_jr, h0, Tuple{Int,Int}[], size(a, 1), 0)
    S = RISDesign.LeastSquaresSystem(G_ej, G_jr, h0, Tuple{Int,Int}[], 1.0, M, nfft)
    Y = RISDesign.apply_F!(similar(S.d), S, a)
    return [sum(S.wgt .* abs2.(view(Y, :, i, j))) / nfft for i in axes(Y, 2), j in axes(Y, 3)]
end

"""Isolation in dB per emitter, as `analysis/metrics.py`: the worst routed target against the
mean energy at receivers more than `excl` positions from every target of that emitter."""
function isolation(E, pairs; excl=2)
    map(axes(E, 1)) do i
        targets = [j + 1 for (j, e) in pairs if e + 1 == i]
        off = [all(abs(j - t) > excl for t in targets) for j in axes(E, 2)]
        minimum(10log10(E[i, t] / (sum(E[i, off]) / count(off))) for t in targets)
    end
end

G_ej, G_jr, h0, pairs = select_inputs(get(ARGS, 1, ""))
design(G_ej, G_jr, h0, pairs, C; maxiter=2)                  # compile both before timing
design_lsqr(G_ej, G_jr, h0, pairs, C; maxiter=2)

ok = true
println("iterations   ||a_lsqr - a_cg|| / ||a_cg||   isolation CG (dB)   isolation LSQR (dB)")
for it in ITERS
    a_cg = design(G_ej, G_jr, h0, pairs, C; maxiter=it)
    a_ls = design_lsqr(G_ej, G_jr, h0, pairs, C; maxiter=it)
    rel = norm(a_ls - a_cg) / norm(a_cg)
    iso_cg = isolation(energies(G_ej, G_jr, h0, a_cg), pairs)
    iso_ls = isolation(energies(G_ej, G_jr, h0, a_ls), pairs)
    fmt(v) = join((@sprintf("%.3f", x) for x in v), " / ")
    @printf("%10d   %28.2e   %17s   %19s\n", it, rel, fmt(iso_cg), fmt(iso_ls))
    global ok &= maximum(abs.(iso_ls - iso_cg)) < 0.01 && (it > 20 || rel < 1e-10)
end

t_cg = minimum(@elapsed(design(G_ej, G_jr, h0, pairs, C; maxiter=100)) for _ in 1:5)
t_ls = minimum(@elapsed(design_lsqr(G_ej, G_jr, h0, pairs, C; maxiter=100)) for _ in 1:5)
@printf("100 iterations on %d threads: normal equations %.3f s, LSQR %.3f s (%.0f x)\n",
        Threads.nthreads(), t_cg, t_ls, t_ls / t_cg)
println(ok ? "PASS" : "FAIL")
exit(ok ? 0 : 1)
