# Design one routing filter and write it to disk.
#
#     julia -t 6 examples/run_design.jl [input_dir] [output.bin]
#
# `input_dir` holds the three .bin inputs of `load_inputs`, of any size: the shapes are read
# off the file names, so a measured set and one written by `test/make_synthetic.py` are both
# accepted.  Without it -- no argument, or a directory that does not exist -- the example falls
# back to a small synthetic set, so the repository runs end to end with no data at all.  The
# filter is written as raw Float64; for the (M, K) Julia matrix that is the C-order (K, M)
# array `np.fromfile(path).reshape(K, M)` reads back.

include(joinpath(@__DIR__, "..", "src", "RISDesign.jl"))
using .RISDesign

include(joinpath(@__DIR__, "inputs.jl"))       # synthetic_inputs, shape_of, default_pairs

dir = get(ARGS, 1, "")
out = get(ARGS, 2, "filter.bin")

G_ej, G_jr, h0, pairs = select_inputs(dir)

# `c = 0.1` weights every (emitter, receiver) combination other than the routed pairs.

elapsed = @elapsed a = design(G_ej, G_jr, h0, pairs, 0.1; maxiter=100)
write(out, a)
println("designed $(size(a, 1)) taps x $(size(a, 2)) transceivers in " *
        "$(round(elapsed, digits=2)) s on $(Threads.nthreads()) threads")
println("wrote $(abspath(out))")
