# Design one routing filter and write it to disk.
#
#     julia -t 6 examples/run_design.jl [input_dir] [output.bin]
#
# `input_dir` holds the three .bin inputs of `load_inputs` (the measured set: 2 emitters,
# 28 transceivers, 101 receivers, N = 7680).  Without it -- no argument, or a directory that
# does not exist -- the example falls back to a small synthetic set, so the repository runs
# end to end with no data at all.  The filter is written as raw Float64; for the (M, K) Julia
# matrix that is the C-order (K, M) array `np.fromfile(path).reshape(K, M)` reads back.

include(joinpath(@__DIR__, "..", "src", "RISDesign.jl"))
using .RISDesign, Random

"""Stand-in Green's functions: exponentially decaying noise, the crudest thing that still
looks like a reverberant medium (a dense tail, no direct path).  Small on purpose -- the cost
of `design` grows with N.  The target `h0`, of the required length 4N-3, is one
Gaussian-windowed sine burst: the pulse each routed receiver should hear."""
function synthetic_inputs(; N=512, K=8, n_em=2, n_rec=16, seed=20250916)
    rng = MersenneTwister(seed)
    legs(m) = randn(rng, N, K, m) .* exp.(-(0:N-1) ./ (N / 6))
    t = 0:(4N - 4)
    return legs(n_em), legs(n_rec), @.(exp(-((t - 220) / 40)^2) * sinpi(0.24 * (t - 220)))
end

dir = get(ARGS, 1, "")
out = get(ARGS, 2, "filter.bin")

# Routes are 0-based (receiver, emitter) pairs: emitter 0 is to be heard at one receiver
# only, emitter 1 at another.  `c = 0.1` weights every other (emitter, receiver) combination.
G_ej, G_jr, h0, pairs = if isdir(dir)
    println("inputs: $dir")
    (load_inputs(dir)..., [(19, 0), (79, 1)])
else
    println("inputs: synthetic" * (isempty(dir) ? "" : " ($dir does not exist)"))
    (synthetic_inputs()..., [(3, 0), (11, 1)])
end

elapsed = @elapsed a = design(G_ej, G_jr, h0, pairs, 0.1; maxiter=100)
write(out, a)
println("designed $(size(a, 1)) taps x $(size(a, 2)) transceivers in " *
        "$(round(elapsed, digits=2)) s on $(Threads.nthreads()) threads")
println("wrote $(abspath(out))")
