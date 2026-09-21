# ---------------------------------------------------- the same design, without the normal equations
#
# Included by RISDesign.jl, inside the module: it reuses `problem_size`, `leg_spectra`, the
# threading helpers and `column_dot!`.

"""
    design_lsqr(G_ej, G_jr, h0, pairs, c; maxiter=100, M=0, Nfft=0) -> a

The filter of `design`, computed without forming the normal equations: LSQR (Paige & Saunders,
1982) on the weighted least-squares system itself,

    F a = d,    (F a)_ij = sqrt(w_ij) * sum_k g_ikj * a_k,    d_ij = sqrt(w_ij) * h_ref_ij,

which only ever applies `F` and its adjoint and so never squares the condition number.  From
`a = 0`, LSQR produces in exact arithmetic the same iterates as CG on `U a = b`, so every argument,
`maxiter` included, means what it means for `design`.

What differs is the cost.  Each iteration applies `F` to all `n_e x n_r` (emitter, receiver)
pairs and `F'` back, reading every receiver leg on the fine grid, where `design` applies a
`K x K` operator in which the receivers have been summed once and for all.

Residuals are held as spectra on the rfft grid of `Nfft`, where nothing wraps.  With the Parseval
weights `(1, 2, ..., 2, 1) / Nfft` their norm is the time-domain least-squares norm, and `irfft`
is the exact adjoint of `rfft` under it.
"""
function design_lsqr(G_ej::Array{Float64,3}, G_jr::Array{Float64,3}, h0::Vector{Float64},
                     pairs, c::Real; maxiter::Int=100, M::Int=0, Nfft::Int=0)
    _, M, nfft = problem_size(G_ej, G_jr, h0, pairs, M, Nfft)
    FFTW.set_num_threads(1)
    return lsqr(LeastSquaresSystem(G_ej, G_jr, h0, pairs, Float64(c), M, nfft), maxiter)
end

"""
    LeastSquaresSystem(G_ej, G_jr, h0, pairs, c, M, nfft)

`F` and `d` of `design_lsqr`: the leg spectra of every emitter and *every* receiver on the fine
grid, the square roots `sw` of the pair weights, the weighted target spectra `d[f, i, j]` (zero
off the routed pairs), and the work arrays of `apply_F!` and `apply_Ft!`.
"""
struct LeastSquaresSystem
    Ge::Array{ComplexF64,3}          # (nf, K, n_e)
    Gj::Array{ComplexF64,3}          # (nf, K, n_r)
    sw::Matrix{Float64}              # (n_e, n_r)
    d::Array{ComplexF64,3}           # (nf, n_e, n_r)
    wgt::Vector{Float64}             # Parseval weights
    M::Int
    Nfft::Int
    Pfull::Matrix{Float64}           # a filter, zero-padded to Nfft
    Zfull::Matrix{Float64}           # an inverse transform, of which the first M rows are kept
    Ahat::Matrix{ComplexF64}         # (nf, K)
    B::Array{ComplexF64,3}           # (nf, n_e, K)
    forward::Any
    inverse::Any
end

function LeastSquaresSystem(G_ej::Array{Float64,3}, G_jr::Array{Float64,3}, h0::Vector{Float64},
                            pairs, c::Float64, M::Int, nfft::Int)
    Ge, Gj = leg_spectra(G_ej, nfft), leg_spectra(G_jr, nfft)
    nf, K, nE = size(Ge)
    nR = size(Gj, 3)
    sw = fill(sqrt(c), nE, nR)
    H0 = rfft(vcat(h0, zeros(nfft - length(h0))))
    d = zeros(ComplexF64, nf, nE, nR)
    for (j, i) in pairs
        sw[i+1, j+1] = 1.0
        @views d[:, i+1, j+1] .= H0
    end
    wgt = fill(2.0, nf)
    wgt[1] = wgt[end] = 1.0                         # DC and, nfft being even, Nyquist
    Ahat = Matrix{ComplexF64}(undef, nf, K)
    return LeastSquaresSystem(Ge, Gj, sw, d, wgt, M, nfft, zeros(nfft, K), zeros(nfft, K), Ahat,
                              Array{ComplexF64,3}(undef, nf, nE, K), plan_rfft(zeros(nfft)),
                              plan_brfft(view(Ahat, :, 1), nfft;
                                         flags=FFTW.ESTIMATE | FFTW.DESTROY_INPUT))
end

"""
    apply_F!(Y, S, a) -> Y

`Y[f, i, j] = sw_ij * sum_k Ge[f,k,i] * Gj[f,k,j] * rfft(a_k)(f)`.  Threads split the frequencies;
every array is walked with unit stride in `f`, and a receiver leg column read for one emitter is
still in cache for the other.
"""
function apply_F!(Y::Array{ComplexF64,3}, S::LeastSquaresSystem, a::AbstractMatrix{Float64})
    nf, K, nE = size(S.Ge)
    pcolumns(K) do k
        @views S.Pfull[1:S.M, k] .= a[:, k]         # the tail of Pfull stays zero
    end
    transform_columns!(S.Ahat, S.forward, S.Pfull)
    pforeach(nf) do fs
        @inbounds for k in 1:K, i in 1:nE, f in fs
            S.B[f, i, k] = S.Ge[f, k, i] * S.Ahat[f, k]
        end
        @inbounds for j in axes(S.Gj, 3)
            for i in 1:nE, f in fs
                Y[f, i, j] = zero(ComplexF64)
            end
            for k in 1:K, i in 1:nE
                @simd for f in fs
                    Y[f, i, j] += S.B[f, i, k] * S.Gj[f, k, j]
                end
            end
            for i in 1:nE
                s = S.sw[i, j]
                @simd for f in fs
                    Y[f, i, j] *= s
                end
            end
        end
    end
    return Y
end

"""
    apply_Ft!(g, S, Y) -> g

The adjoint: `g[:, k] =` first `M` samples of `irfft(sum_ij sw_ij conj(Ge[:,k,i] Gj[:,k,j]) Y[:,i,j])`.
The receiver sum runs in the same order for every frequency, whatever the thread count.
"""
function apply_Ft!(g::AbstractMatrix{Float64}, S::LeastSquaresSystem, Y::Array{ComplexF64,3})
    nf, K, nE = size(S.Ge)
    pforeach(nf) do fs
        @inbounds for k in 1:K, i in 1:nE, f in fs
            S.B[f, i, k] = zero(ComplexF64)
        end
        @inbounds for j in axes(S.Gj, 3), k in 1:K, i in 1:nE
            s = S.sw[i, j]
            @simd for f in fs
                S.B[f, i, k] += conj(S.Gj[f, k, j]) * (s * Y[f, i, j])
            end
        end
        @inbounds for k in 1:K, f in fs
            acc = zero(ComplexF64)
            for i in 1:nE
                acc += conj(S.Ge[f, k, i]) * S.B[f, i, k]
            end
            S.Ahat[f, k] = acc / S.Nfft             # brfft leaves out the 1/Nfft
        end
    end
    transform_columns!(S.Zfull, S.inverse, S.Ahat)  # consumes Ahat
    pcolumns(K) do k
        @views g[:, k] .= S.Zfull[1:S.M, k]
    end
    return g
end

"""
    combine!(u, x, alpha, S) -> ||x - alpha*u||

`u = x - alpha * u` in place, returning the time-domain norm of the result; the norm is summed
per receiver and then over the receivers in order.
"""
function combine!(u::Array{ComplexF64,3}, x::Array{ComplexF64,3}, alpha::Float64,
                  S::LeastSquaresSystem)
    nf, nE, nR = size(u)
    part = zeros(nR)
    pforeach(nR) do js
        @inbounds for j in js
            acc = 0.0
            for i in 1:nE
                @simd for f in 1:nf
                    u[f, i, j] = x[f, i, j] - alpha * u[f, i, j]
                    acc += S.wgt[f] * abs2(u[f, i, j])
                end
            end
            part[j] = acc
        end
    end
    return sqrt(sum(part) / S.Nfft)
end

"""
    lsqr(S, maxiter) -> a

LSQR from `a = 0`, exactly `maxiter` iterations and no stopping test, as for `design`:
Golub-Kahan bidiagonalisation `beta u = F v - alpha u`, `alpha v = F' u - beta v`, and the
Givens-rotation update of the solution along `w`.
"""
function lsqr(S::LeastSquaresSystem, maxiter::Int)
    K = size(S.Ge, 2)
    a, v, w, g = (zeros(S.M, K) for _ in 1:4)
    part = zeros(K)
    u, Fv = zero(S.d), similar(S.d)
    beta = combine!(u, S.d, 0.0, S)                 # u = d
    u ./= beta
    apply_Ft!(v, S, u)
    alpha = sqrt(column_dot!(part, v, v))
    v ./= alpha
    copyto!(w, v)
    phibar, rhobar = beta, alpha
    for _ in 1:maxiter
        beta = combine!(u, apply_F!(Fv, S, v), alpha, S)
        u ./= beta
        apply_Ft!(g, S, u)
        @. v = g - beta * v
        alpha = sqrt(column_dot!(part, v, v))
        v ./= alpha
        rho = hypot(rhobar, beta)
        cs, sn = rhobar / rho, beta / rho
        theta, rhobar = sn * alpha, -cs * alpha
        phi, phibar = cs * phibar, sn * phibar
        @. a += (phi / rho) * w
        @. w = v - (theta / rho) * w
    end
    return a
end
