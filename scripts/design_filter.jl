# Design one filter from a Green's-function file and write it as a (transceivers, taps) .npy.
#
#   julia -t 6 --project=. scripts/design_filter.jl --greens Greens.npz --pairs "19,0;79,1" \
#         --c 0.1 --maxiter 100 --M 10000 --delay 8000 --out filter.npy
#
# --greens   .npz holding G_ej (emitters x transceivers x samples) and G_jr (receivers x ...)
# --N        samples of each leg kept, from the start                          (default 7680)
# --pairs    routed 0-based (receiver, emitter) pairs, ";"-separated
# --c        weight of every unrouted (emitter, receiver) combination
# --maxiter  conjugate-gradient iterations                                     (default 100)
# --M        filter taps                                                       (default 2N-1)
# --delay    sample at which the target pulse peaks; keep it below about M     (default M-1)
# --band     passband of the target in Hz, "low,high"                          (default 120,12000)
# --alpha    Tukey window parameter of the target                              (default 0.3)
# --fs       sampling rate in Hz                                               (default 44100)
# --normalize  "spectrum" scales the filter so max over transceivers and frequencies of
#            |rfft(filter)| is 1, the convention of the lab's filter files        (default none)
# --out      output .npy, shape (transceivers, M), float64

include(joinpath(@__DIR__, "..", "src", "RISDesign.jl"))
using .RISDesign, NPZ, FFTW

function parse_args(args)
    length(args) % 2 == 0 || error("arguments come in --key value pairs")
    opts = Dict(args[i][3:end] => args[i+1] for i in 1:2:length(args))
    for k in ("greens", "pairs", "c", "out")
        haskey(opts, k) || error("missing --$k")
    end
    return opts
end

"""
Tukey-windowed band-pass FIR of `2delay + 1` taps, peaking at sample `delay`, unit gain at the
centre of the passband.  The same construction as scipy.signal.firwin(2delay+1, band,
pass_zero=False, window=("tukey", alpha), fs=fs, scale=True).
"""
function target_fir(delay, f_lo, f_hi, alpha, fs)
    n = 2delay + 1
    m = (0:n-1) .- delay
    lo, hi = f_lo / (fs / 2), f_hi / (fs / 2)
    h = hi .* sinc.(hi .* m) .- lo .* sinc.(lo .* m)
    width = floor(Int, alpha * (n - 1) / 2)
    w = ones(n)
    for i in 0:width
        w[i+1] = 0.5 * (1 + cos(pi * (-1 + 2i / alpha / (n - 1))))
        w[n-i] = w[i+1]
    end
    h .*= w
    return h ./ sum(h .* cos.(pi .* m .* (lo + hi) / 2))
end

opts = parse_args(ARGS)
g = npzread(opts["greens"])
N = parse(Int, get(opts, "N", "7680"))
G_ej = permutedims(g["G_ej"][:, :, 1:N], (3, 2, 1))       # (samples, transceivers, emitters)
G_jr = permutedims(g["G_jr"][:, :, 1:N], (3, 2, 1))
M = parse(Int, get(opts, "M", string(2N - 1)))
delay = parse(Int, get(opts, "delay", string(M - 1)))
f_lo, f_hi = parse.(Float64, split(get(opts, "band", "120,12000"), ","))
pairs = [Tuple(parse.(Int, split(p, ","))) for p in split(opts["pairs"], ";")]

h = target_fir(delay, f_lo, f_hi, parse(Float64, get(opts, "alpha", "0.3")),
               parse(Float64, get(opts, "fs", "44100")))
h0 = zeros(M + 2N - 2)
length(h) <= length(h0) || error("--delay too large for this M and N")
h0[1:length(h)] .= h

a = design(G_ej, G_jr, h0, pairs, parse(Float64, opts["c"]);
           maxiter=parse(Int, get(opts, "maxiter", "100")), M=M)
normalization = get(opts, "normalize", "none")
normalization in ("none", "spectrum") || error("--normalize must be none or spectrum")
normalization == "spectrum" && (a ./= maximum(abs, rfft(a, 1)))
npzwrite(opts["out"], permutedims(a))
println("wrote ", opts["out"], "  (", size(a, 2), " transceivers x ", size(a, 1), " taps)")
