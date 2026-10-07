# The solver results of the paper:  julia -t 6 --project=. reproduce.jl
include("RISDesign.jl")
using NPZ, Printf, Statistics

load(file, keys...) = npzread(joinpath(@__DIR__, "data", file), keys...)
legs(files...) = permutedims(vcat(load.(files)...), (3, 2, 1))   # (N, K, n_E or n_R)
G_ej, G_jr = legs("G_ej.npy"), legs("G_jr_1.npy", "G_jr_2.npy")
(N, K, nE), nR = size(G_ej), size(G_jr, 3)
pairs(file) = Tuple.(eachrow(load(file, ["pairs"])["pairs"]))
c, M, n = 0.1, 10000, fftlen(10000, N)
# target: Tukey (0.3)-windowed band-pass FIR, 120 Hz - 12 kHz, 2 * 6000 + 1 taps peaking at 6000
t, lo, hi = -6000:6000, 120 / 22050, 12000 / 22050
h = (hi .* sinc.(hi .* t) .- lo .* sinc.(lo .* t)) .*
    [(i = 6000 - abs(m)) <= 1800 ? 0.5 * (1 + cos(π * (-1 + 2i / 0.3 / 12000))) : 1.0 for m in t]
h ./= sum(h .* cos.(π .* t .* (lo + hi) / 2))

for file in ("single.npz", "double.npz")
    a = design(G_ej, G_jr, h, pairs(file), c, M, 200)[1]
    println("Figs. 2, 3: optimized lab filter of $file regenerated bit for bit: ",
            permutedims(a ./ maximum(abs, rfft(a, 1))) == load(file, ["opt_filter"])["opt_filter"])
end

P = pairs("single.npz")                                   # Fig. 2 from here on
t_cg = median(@elapsed(design(G_ej, G_jr, h, P, c, M, 200)) for _ in 1:7)

# Crosstalk (dB) of each routed pair (j, i): mean energy of emitter i at the receivers more than
# 2 positions from j, relative to its energy at j (model responses).
function crosstalk(a)
    p, F! = lsq_operator(G_ej, G_jr, ones(nE, nR), M, n)
    E = (p .= a; sum([1; fill(2, n ÷ 2 - 1); 1] .* abs2.(F!()); dims=1))   # by Parseval
    [10log10(mean(E[1, i+1, abs.((0:nR-1) .- j) .> 2]) / E[1, i+1, j+1]) for (j, i) in P]
end
a60, l60 = design(G_ej, G_jr, h, P, c, M, 60)[1], design_lsqr(G_ej, G_jr, h, P, c, M, 60)
(a, α, β, R), l = design(G_ej, G_jr, h, P, c, M, 200), design_lsqr(G_ej, G_jr, h, P, c, M, 200)
t_ls = median(@elapsed(design_lsqr(G_ej, G_jr, h, P, c, M, 200)) for _ in 1:3)
@printf("LSQR: |a_lsqr - a_cg| / |a_cg| = %.1e after 60, %.1e after 200 iterations\n",
        norm(l60 - a60) / norm(a60), norm(l - a) / norm(a))
@printf("LSQR: max crosstalk difference %.3f dB\n", maximum(abs, crosstalk(a) - crosstalk(l)))
@printf("Intro/LSQR: 200 iterations %.2f s (CG) vs %.1f s (LSQR)\n", t_cg, t_ls)

# Conditioning. The eigenvalues θ of the Lanczos matrix T_200 of CG lie in [λ_min(U), λ_max(U)],
# so λ_max(U) >= θ_max. λ_min(U) is at most the Rayleigh quotient of any filter, here of the real
# part of the Rayleigh-Ritz vector on hann(m) exp(i(ω_c + 2πl/M)m) e_k, l = -10..10, every k,
# with ω_c the bin where the smallest eigenvalue of the symbol of U is lowest.
θ = eigvals(SymTridiagonal(1 ./ α .+ [0; β ./ α][1:end-1], sqrt.(β[1:end-1]) ./ α[1:end-1]))
symbol(f) = f > n ÷ 2 ? conj(symbol(n - f)) :            # of U at 0-based bin f of a length-n fft
    Hermitian([k <= s ? R[packed(k, s), f+1] : 0.0im for k in 1:K, s in 1:K])
fc, m = argmin(f -> eigmin(symbol(f)), 0:n÷2), 0:M-1
V = (0.5 .- 0.5cos.(2π * m / (M - 1))) .* cis.((2π * fc / n .+ 2π * (-10:10)' / M) .* m)
V̂ = fft([V; zeros(n - M, 21)], 1)                        # V' U V over the bins that hold V
B = sum(kron(symbol(f), conj(V̂[f+1, :]) * transpose(V̂[f+1, :])) for f in fc-60:fc+60)
x = V * reshape(eigvecs(Hermitian(B), Hermitian(kron(I(K), V' * V)))[:, 1], 21, K)
p, Up, U! = matvec(R, n, M, K)
λmin = (p .= real.(x); U!(); dot(p, Up) / dot(p, p))
@printf("Conditioning: kappa(T_200) = %.1e, kappa(U) >= %.1e\n", θ[end] / θ[1], θ[end] / λmin)
