"""
    RISDesign

The solver behind the paper's filters, as one straight line of named steps, each a small
function whose docstring states the formula it evaluates.  `reference/reference_design.py` is
the literal, unoptimised transcription of the same equations; `test/verify.py` checks the two
against each other, and everything below called "the reference" is that file.

    include("src/RISDesign.jl"); using .RISDesign
    G_ej, G_jr, h0 = load_inputs("inputs")
    a = design(G_ej, G_jr, h0, [(19, 0), (79, 1)], 0.1)      # (15359, 28) Matrix{Float64}

Transceiver `k` replays the filter `a_k`, and emitter `i` reaches receiver `j` through the
composite `g_ikj = G_ej[i,k] * G_jr[j,k]` (`*` = convolution, length `conv_len = 2N-1`).  With
`w_ij = 1` on the routed pairs and `c` on every other (emitter, receiver) combination, and the
target `h_ref_ij = h0` on the routed pairs and `0` elsewhere, the design minimises

    sum_ij w_ij || sum_k g_ikj * a_k - h_ref_ij ||^2    over the M = conv_len taps of each a_k

by plain conjugate gradients on the normal equations `U a = b`, from `a = 0`, for a fixed number
of iterations.  Per bin `f` of the rfft grid of length `Nfft`, with `ghat` the spectrum of `g`:

    R_ks(f) = sum_ij w_ij conj(ghat_ikj(f)) ghat_isj(f)     the operator, Hermitian in (k, s)
    (U a)_k = first M samples of irfft(sum_s R_ks * rfft(a_s, Nfft))
    b[n, k] = sum over routed pairs of sum_m g_ikj(m) h0(n + m),   n = 0 .. M-1

Same algorithm as the reference -- same operator, same right-hand side, plain CG from a zero
start, Float64 throughout.  Four things make it fast, each marked FAST (n) in the docstring of
the function that implements it (the sizes quoted are the paper's, N = 7680, K = 28, 2 emitters
and 101 receivers):

1. the FFT grid is 30720, not the reference's 30717 (`design`),
2. `R` is factorised, so no 101 x 28 composite spectra are ever formed (`operator`),
3. the sum over the 101 receivers runs on a half-length grid (`gram_matrix`, `to_fine_grid`),
4. `R(f)` is Hermitian, so only its upper triangle is stored and applied (`apply_operator!`).

The filter does not depend on the number of threads: every sum that is split across tasks is
split one filter column per task, and the columns are then added up in a fixed order.
"""
module RISDesign

using FFTW, LinearAlgebra, Base.Threads

export design, load_inputs

# ------------------------------------------------------------------------------------ helpers

"""
    pforeach(f, n; npiece)

Call `f(range)` on contiguous pieces of `1:n`, one Julia task per piece, and wait.  Having more
pieces than threads lets the scheduler balance the machine's performance and efficiency cores.
The default `npiece` follows `nthreads()`, so a loop reducing across pieces would depend on the
thread count -- none does.
"""
function pforeach(f, n::Int; npiece::Int=min(n, 4 * nthreads()))
    @sync for p in 1:npiece
        Threads.@spawn f((((p - 1) * n) ÷ npiece + 1):((p * n) ÷ npiece))
    end
end

"call `f(k)` on every column index `k` in `1:K`, one task per column"
pcolumns(f, K::Int) = pforeach(cols -> foreach(f, cols), K; npiece=K)

"index of entry `(k, s)`, `k <= s`, in the packed upper triangle of a `K x K` Hermitian matrix"
@inline packed(k::Int, s::Int) = (s * (s - 1)) ÷ 2 + k

"""
    transform_columns!(out, plan, input) -> out

Apply the one-dimensional FFTW `plan` to every column of `input`, one column per task.  Every
transform here is threaded over columns like this rather than by an FFTW batched plan, because
the batched complex-to-real plan does not thread at all.  The plan is built for one column and
used on all of them, so every column must start in the same FFTW alignment class:
that is why `design` requires an even `Nfft`, which puts the columns of the real work arrays a
multiple of 16 bytes apart.
"""
function transform_columns!(out::AbstractMatrix, plan, input::AbstractMatrix)
    pcolumns(size(input, 2)) do col
        @views mul!(out[:, col], plan, input[:, col])
    end
    return out
end

"""
    load_inputs(dir; N=7680, K=28, n_e=2, n_r=101) -> (G_ej, G_jr, h0)

Read the three raw Float64 dumps of `dir` in host byte order (little-endian here), the layout
`test/make_synthetic.py` also writes:

    G_ej_{n_e}x{K}x{N}_f64le.bin     emitter legs,  C order (n_e, K, N), read as `[n,k,i]`
    G_jr_{n_r}x{K}x{N}_f64le.bin     receiver legs, C order (n_r, K, N), read as `[n,k,j]`
    target_h0_len{4N-3}_f64le.bin    the design target `h0`

The defaults are the sizes of the paper's measured set; the numbers in the file names are the
shapes, so any other set is read by passing its own.
"""
function load_inputs(dir::AbstractString; N::Int=7680, K::Int=28, n_e::Int=2, n_r::Int=101)
    legs(name) = (p = joinpath(dir, name);
                  read!(p, Array{Float64}(undef, N, K, filesize(p) ÷ (8 * N * K))))
    return (legs("G_ej_$(n_e)x$(K)x$(N)_f64le.bin"), legs("G_jr_$(n_r)x$(K)x$(N)_f64le.bin"),
            read!(joinpath(dir, "target_h0_len$(4N-3)_f64le.bin"), Vector{Float64}(undef, 4N - 3)))
end

# ------------------------------------------------------------------------------ 1: leg spectra

"""
    leg_spectra(G, Nfft) -> Ghat

`Ghat[f, k, m] = rfft(G[:, k, m], Nfft)[f]`: every length-`N` Green's-function leg of
`G[n, k, m]`, zero-padded to `Nfft` and transformed.
"""
function leg_spectra(G::Array{Float64,3}, Nfft::Int)
    N, K, m = size(G)
    legs = reshape(G, N, K * m)
    Ghat = Matrix{ComplexF64}(undef, Nfft ÷ 2 + 1, K * m)
    plan = plan_rfft(zeros(Nfft))
    pforeach(K * m) do cols
        padded = zeros(Nfft)                            # the zero tail is the zero padding
        for col in cols
            copyto!(padded, 1, legs, (col - 1) * N + 1, N)
            @views mul!(Ghat[:, col], plan, padded)
        end
    end
    return reshape(Ghat, :, K, m)
end

# --------------------------------------------------------------------------- 2: receiver Gram

"""
    gram_matrix(Gj) -> S

`S[f, packed(k,s)] = sum_j conj(Gj[f,k,j]) * Gj[f,s,j]` for `k <= s`: the `K x K` Hermitian Gram
matrix of the receiver legs, summed over *all* receivers, kept as its upper triangle.  The legs
of one frequency are gathered receiver-major first, so that the sum runs over adjacent memory.

FAST (3): this is the largest sum of the design, and `to_fine_grid` halves it.
"""
function gram_matrix(Gj::Array{ComplexF64,3})
    nf, K, nR = size(Gj)
    S = Matrix{ComplexF64}(undef, nf, packed(K, K))
    pforeach(nf) do freqs
        leg = Matrix{ComplexF64}(undef, nR, K)          # all legs at one frequency
        @inbounds for f in freqs
            for k in 1:K, j in 1:nR
                leg[j, k] = Gj[f, k, j]
            end
            for s in 1:K, k in 1:s
                acc = zero(ComplexF64)
                @simd for j in 1:nR                     # @simd reassociates this sum
                    acc += conj(leg[j, k]) * leg[j, s]
                end
                S[f, packed(k, s)] = acc
            end
        end
    end
    return S
end

"""
    to_fine_grid(S, Nc, Nfft, maxlag) -> Sf

Move the Gram from the coarse grid `Nc` to the fine grid `Nfft`: transform back to the
correlation lags, which vanish for `|m| > maxlag`, and transform that same lag sequence on
`Nfft`.  Exact whenever `Nc >= 2*maxlag + 1`.  Consumes `S`.

FAST (3): `S(f)` is a sum of cross-spectra of two length-`N` legs, so its lags obey `|m| <= N-1`
and the receiver sum may run on `Nc = 15360` instead of the fine grid.  These few lines halve
both that sum and the 2828 receiver-leg transforms feeding it.
"""
function to_fine_grid(S::Matrix{ComplexF64}, Nc::Int, Nfft::Int, maxlag::Int)
    Sf = Matrix{ComplexF64}(undef, Nfft ÷ 2 + 1, size(S, 2))
    back = plan_brfft(view(S, :, 1), Nc; flags=FFTW.ESTIMATE | FFTW.DESTROY_INPUT)
    ahead = plan_rfft(zeros(Nfft))
    pforeach(size(S, 2)) do entries
        lags = Vector{Float64}(undef, Nc)               # lag m in entry m+1, negative m wrapped
        padded = zeros(Nfft)                            # only the two lag windows are ever set
        for e in entries
            @views mul!(lags, back, S[:, e])            # brfft leaves out the 1/Nc, hence ./ Nc
            padded[1:maxlag+1] .= lags[1:maxlag+1] ./ Nc                # lags 0 .. maxlag
            padded[Nfft-maxlag+1:Nfft] .= lags[Nc-maxlag+1:Nc] ./ Nc    # lags -maxlag .. -1
            @views mul!(Sf[:, e], ahead, padded)
        end
    end
    return Sf
end

# -------------------------------------------------------------------------------- 3: operator

"""
    operator(Ge, Gt, S, routes, c, Nfft) -> R

`R[packed(k,s), f] = sum_ij w_ij conj(ghat_ikj(f)) * ghat_isj(f)` for `k <= s`: the upper
triangle of the Hermitian normal-equation operator, with composite spectra
`ghat_ikj = ge_ik * gj_jk` and weights `w_ij = 1` on the routed pairs, `c` elsewhere.

FAST (2): writing `w` as `c` everywhere plus `1-c` on the routed pairs,

    R_ks = c * E_ks * S_ks + (1 - c) * sum over routes conj(t_ik) * t_is,  t_ik = ge_ik * gj_jk
    E_ks = sum_i conj(ge_ik) * ge_is

so only the 2 emitter legs, the 2 routed receiver legs and `S` are needed, and the reference's
101 x 28 composite spectra are never formed.  `1/Nfft`, the normalisation of the inverse
transform in the matvec, is folded in here.  `R` keeps the frequency last, unlike the spectra,
because `apply_operator!` sweeps the packed triangle of one frequency at a time.

Storing one triangle also makes `R` exactly Hermitian, where the reference computes both
triangles independently and is Hermitian only to round-off -- one of the small differences that
the chaotic tail of the CG recursion later amplifies (the note `test/verify.py` prints says how
much, and why the 100-iteration filters are compared through the physics instead).
"""
function operator(Ge::Array{ComplexF64,3}, Gt::Array{ComplexF64,3}, S::Matrix{ComplexF64},
                  routes::Vector{Tuple{Int,Int}}, c::Float64, Nfft::Int)
    nf, K, nE = size(Ge)
    R = Matrix{ComplexF64}(undef, size(S, 2), nf)
    pforeach(nf) do freqs
        t = Matrix{ComplexF64}(undef, K, length(routes))
        @inbounds for f in freqs
            for (q, (jt, ie)) in enumerate(routes), k in 1:K
                t[k, q] = Ge[f, k, ie] * Gt[f, k, jt]               # routed composite spectra
            end
            for s in 1:K, k in 1:s
                E = sum(conj(Ge[f, k, i]) * Ge[f, s, i] for i in 1:nE)
                T = sum(conj(t[k, q]) * t[s, q] for q in axes(t, 2))
                e = packed(k, s)
                R[e, f] = (c * E * S[f, e] + (1 - c) * T) / Nfft
            end
        end
    end
    return R
end

# ------------------------------------------------------------------------- 4: right-hand side

"""
    rhs(Ge, Gt, h0, routes, Nfft, M) -> b

`b[n, k] = sum over routes of sum_m g_ikj(m) * h0(n + m)`, `n = 0 .. M-1`: the correlation of
the routed composite responses with the design target, evaluated as
`irfft(sum over routes of conj(ghat_ikj) * rfft(h0, Nfft))[1:M, :]`.  That is this correlation
and not a circular one because the lags obey `n + m <= (M-1) + (conv_len-1) < Nfft`.
"""
function rhs(Ge::Array{ComplexF64,3}, Gt::Array{ComplexF64,3}, h0::Vector{Float64},
             routes::Vector{Tuple{Int,Int}}, Nfft::Int, M::Int)
    nf, K, _ = size(Ge)
    H0 = rfft(vcat(h0, zeros(Nfft - length(h0))))
    B = zeros(ComplexF64, nf, K)
    @inbounds for (jt, ie) in routes, k in 1:K
        @simd for f in 1:nf
            B[f, k] += conj(Ge[f, k, ie] * Gt[f, k, jt]) * H0[f]
        end
    end
    return irfft(B, Nfft, 1)[1:M, :]
end

# --------------------------------------------------------------------- 5: conjugate gradients

"""
    apply_operator!(Y, P, R)

`Y[f, k] = sum_s R_ks(f) * P[f, s]`, with `R` stored as its packed upper triangle:
`R_ks = R[packed(k,s), f]` for `k <= s`, and `R_sk = conj(R_ks)` because `R(f)` is Hermitian.

FAST (4): each stored entry is therefore used twice, once for `y_k += R_ks p_s` and once for
`y_s += conj(R_ks) p_k`, and the CG loop streams half the memory on each of its iterations.
This is the hottest loop of the design; `@simd` reassociates its inner sum.
"""
function apply_operator!(Y::Matrix{ComplexF64}, P::Matrix{ComplexF64}, R::Matrix{ComplexF64})
    nf, K = size(P)
    size(Y) == size(P) && size(R) == (packed(K, K), nf) || error("apply_operator!: size mismatch")
    pforeach(nf) do freqs
        p = Vector{ComplexF64}(undef, K)
        y = Vector{ComplexF64}(undef, K)
        @inbounds for f in freqs
            for k in 1:K
                p[k] = P[f, k]
                y[k] = zero(ComplexF64)
            end
            e = 0                                       # e + k == packed(k, s) inside the loop
            for s in 1:K
                ps = p[s]
                ys = zero(ComplexF64)
                @simd ivdep for k in 1:s-1              # each off-diagonal feeds y[s] and y[k]
                    Rks = R[e+k, f]
                    ys += conj(Rks) * p[k]
                    y[k] += Rks * ps
                end
                y[s] += ys + R[e+s, f] * ps
                e += s
            end
            for k in 1:K
                Y[f, k] = y[k]
            end
        end
    end
end

"""
    column_dot!(part, A, B) -> <A, B>

Dot product of two `(M, K)` arrays, accumulated per column and then summed over the columns in
order, so that the value -- and with it the designed filter -- is the same at any thread count.
The reference instead calls `numpy.vdot` on the flattened array, which sums it pairwise; the two
orders differ by round-off, as any two orders would.
"""
function column_dot!(part::Vector{Float64}, A::AbstractMatrix, B::AbstractMatrix)
    pcolumns(size(A, 2)) do k
        @views part[k] = dot(A[:, k], B[:, k])
    end
    return sum(part)
end

"""
    conjugate_gradient(R, b, Nfft, maxiter) -> a

Plain CG for `U a = b` from `a = 0`, no preconditioner: `alpha = <r,r> / <p,Up>`, `a += alpha*p`,
`r -= alpha*Up`, `beta = <r_new,r_new> / <r,r>`, `p = r + beta*p`.  The matvec is
`(U p)_k = irfft(sum_s R_ks * rfft(p_s, Nfft))[1:M]`, the `1/Nfft` of the inverse transform
already folded into `R`.  There is no stopping test: exactly `maxiter` iterations are run, and
the iteration count is the regularisation -- at the paper's size the relative residual is still
about 0.29 after the 100 iterations the paper uses.
"""
function conjugate_gradient(R::Matrix{ComplexF64}, b::Matrix{Float64}, Nfft::Int, maxiter::Int)
    M, K = size(b)
    a, r = zeros(M, K), copy(b)
    Pfull = zeros(Nfft, K)                    # the direction, zero-padded for the forward FFT
    Zfull = zeros(Nfft, K)                    # U*p, of which only the first M rows are kept
    p, Up = view(Pfull, 1:M, :), view(Zfull, 1:M, :)
    copyto!(p, r)
    Phat = Matrix{ComplexF64}(undef, Nfft ÷ 2 + 1, K)
    Yhat = similar(Phat)
    forward = plan_rfft(view(Pfull, :, 1))
    inverse = plan_brfft(view(Yhat, :, 1), Nfft; flags=FFTW.ESTIMATE | FFTW.DESTROY_INPUT)
    part = zeros(K)
    rs = column_dot!(part, r, r)
    for _ in 1:maxiter
        transform_columns!(Phat, forward, Pfull)
        apply_operator!(Yhat, Phat, R)
        transform_columns!(Zfull, inverse, Yhat)                          # consumes Yhat
        alpha = rs / column_dot!(part, p, Up)
        pcolumns(K) do k
            @views a[:, k] .+= alpha .* p[:, k]
            @views r[:, k] .-= alpha .* Up[:, k]
            @views part[k] = sum(abs2, r[:, k])
        end
        rs_new = sum(part)
        beta, rs = rs_new / rs, rs_new
        pcolumns(K) do k
            @views p[:, k] .= r[:, k] .+ beta .* p[:, k]
        end
    end
    return a
end

# ------------------------------------------------------------------------------------- design

"""
    design(G_ej, G_jr, h0, pairs, c; maxiter=100, M=0, Nfft=0) -> a

The `(M, K)` filter -- `(15359, 28)` at the paper's size -- column `k` being the filter of
transceiver `k`.  `M` is the number of filter taps; `0` means the composite length `2N - 1`.
`h0` is the design target in the time domain and must have length `M + 2N - 2`, the shortest
grid on which neither the matvec nor `b` wraps around; its peak is the arrival time the design
aims for, so it must lie within reach of the filter (in practice before about sample `M`).  `pairs` are the 0-based `(receiver, emitter)` routes and `c` is the weight of
every unrouted (emitter, receiver) combination.  `write(io, a)` reproduces byte for byte the
C-order `(K, M)` array the reference returns, which NumPy reads with `.reshape(K, M)`.

FAST (1): `Nfft` defaults to the smallest 5-smooth length `>= conv_len + M - 1 = 30717`, which is
30720.  Any grid at least that long leaves the matvec and `b` free of wrap-around and so gives
the same operator; the reference uses 30717 itself, whose prime factor 3413 makes its transform
about eight times slower.  Pass `Nfft` to check that claim on another admissible grid -- it must
also be even, which is a constraint of `transform_columns!` and not of the operator.

Sets `FFTW.set_num_threads(1)`, which is process-global: transforms here are threaded over
columns, and threaded FFTW plans inside those tasks would oversubscribe the machine.
"""
function design(G_ej::Array{Float64,3}, G_jr::Array{Float64,3}, h0::Vector{Float64},
                pairs, c::Real; maxiter::Int=100, M::Int=0, Nfft::Int=0)
    N = size(G_ej, 1)
    M = M == 0 ? 2N - 1 : M                         # filter taps; default = composite length
    M >= 1 || error("M must be positive, got $M")
    Nmin = M + (2N - 1) - 1                         # 30717 by default: no wrap-around anywhere
    length(h0) == Nmin || error("h0 must have length $Nmin, got $(length(h0))")
    all(0 <= j < size(G_jr, 3) && 0 <= i < size(G_ej, 3) for (j, i) in pairs) ||
        error("pairs must be 0-based (receiver, emitter) indices")
    allunique(pairs) || error("pairs must be unique: a repeated route would be weighted twice")
    nfft = Nfft == 0 ? nextprod((2, 3, 5), Nmin) : Nfft     # 0 asks for the default grid, 30720
    nfft >= Nmin && iseven(nfft) ||
        error("Nfft must be even and >= $Nmin, got $nfft")   # even: see transform_columns!
    Nc = nextprod((2, 3, 5), 2N - 1)                # 15360: holds the Gram's lags |m| <= N-1
    FFTW.set_num_threads(1)

    routed = unique(j + 1 for (j, i) in pairs)      # the receivers that carry a routed pair
    routes = [(findfirst(==(j + 1), routed)::Int, i + 1) for (j, i) in pairs]   # 1-based

    Ge = leg_spectra(G_ej, nfft)                    # (nf, K, n_e)     emitter legs
    Gt = leg_spectra(G_jr[:, :, routed], nfft)      # (nf, K, routed)  routed receiver legs
    Sc = gram_matrix(leg_spectra(G_jr, Nc))         # sum over all receivers, on the coarse grid
    S = to_fine_grid(Sc, Nc, nfft, N - 1)
    R = operator(Ge, Gt, S, routes, Float64(c), nfft)
    b = rhs(Ge, Gt, h0, routes, nfft, M)
    return conjugate_gradient(R, b, nfft, maxiter)
end

end # module
