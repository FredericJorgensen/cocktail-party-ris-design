# Filter design of the paper: a (M, K) filter, M taps per transceiver, for the Green's functions
# G_ej (N, K, n_E) and G_jr (N, K, n_R). The routed 0-based (receiver, emitter) pairs have target h
# and weight 1, all other pairs target 0 and weight c. design runs CG on the normal equations
# U a = b, design_lsqr LSQR on F a = d, both from a = 0. Parallel sums are added in a fixed order,
# so the filters do not depend on the thread count.
using FFTW, LinearAlgebra

pfor(f, n, m=4Threads.nthreads()) = @sync for p in 1:m   # f(range) on m contiguous pieces of 1:n
    Threads.@spawn f((p - 1) * n ÷ m + 1:p * n ÷ m)
end
pcols(f, K) = pfor(ks -> foreach(f, ks), K, K)        # f(k), one task per column
cols!(Y, plan, X) = (pcols(k -> mul!(view(Y, :, k), plan, view(X, :, k)), size(X, 2)); Y)
psum(f, K) = (s = zeros(K); pcols(k -> s[k] = f(k), K); sum(s))  # f(1) + ... + f(K), in order
coldot(A, B) = psum(k -> @views(dot(A[:, k], B[:, k])), size(A, 2))
packed(k, s) = s * (s - 1) ÷ 2 + k                    # (k, s), k <= s, in a packed upper triangle
fftlen(M, N) = nextprod((2, 3, 5), M + 2N - 2)        # 5-smooth FFT grid without wrap-around
iplan(n) = plan_brfft(zeros(ComplexF64, n ÷ 2 + 1), n) # irfft without its 1/n; destroys its input

# rfft of every leg G[:, k, m], zero-padded to n
function spectra(G, n)
    N, K, m = size(G)
    Ĝ, plan = Matrix{ComplexF64}(undef, n ÷ 2 + 1, K * m), plan_rfft(zeros(n))
    pfor(K * m) do cols
        x = zeros(n)
        foreach(col -> mul!(view(Ĝ, :, col), plan, copyto!(x, 1, G, (col - 1) * N + 1, N)), cols)
    end
    reshape(Ĝ, :, K, m)
end

# Receiver Gram S_ks = sum_j conj(ĝ_jk) ĝ_js (packed, k <= s), summed on the coarse grid nc,
# which holds its lags |m| <= N - 1, and moved to the fine grid n through those lags.
function gram(G_jr, nc, n)
    N, K, nR = size(G_jr)
    Ĝ, ne = spectra(G_jr, nc), packed(K, K)
    S = Matrix{ComplexF64}(undef, nc ÷ 2 + 1, ne)
    pfor(size(S, 1)) do fs
        leg = Matrix{ComplexF64}(undef, nR, K)
        @inbounds for f in fs
            leg .= transpose(view(Ĝ, f, :, :))
            for s in 1:K, k in 1:s
                acc = zero(ComplexF64)
                @simd for j in 1:nR
                    acc += conj(leg[j, k]) * leg[j, s]
                end
                S[f, packed(k, s)] = acc
            end
        end
    end
    L, X = cols!(Matrix{Float64}(undef, nc, ne), iplan(nc), S), zeros(n, ne)   # negative lags wrap
    pcols(e -> @views((X[1:N, e] .= L[1:N, e] ./ nc; X[n-N+2:n, e] .= L[nc-N+2:nc, e] ./ nc)), ne)
    cols!(similar(S, n ÷ 2 + 1, ne), plan_rfft(zeros(n)), X)
end

# U p = first M samples of irfft(R(f) rfft(p)) for the packed Hermitian symbol R (1/n folded in).
# Returns p (M, K), U p and the function that updates U p from p in place.
function matvec(R, n, M, K)
    P, Z, P̂ = zeros(n, K), zeros(n, K), zeros(ComplexF64, n ÷ 2 + 1, K)   # P̂ = rfft(p), then R P̂
    fwd, inv = plan_rfft(zeros(n)), iplan(n)
    function U!()
        cols!(P̂, fwd, P)
        pfor(size(P̂, 1)) do fs
            p, y = zeros(ComplexF64, K), zeros(ComplexF64, K)
            @inbounds for f in fs
                p .= view(P̂, f, :)
                for s in 1:K
                    ps, ys = p[s], zero(ComplexF64)
                    @simd ivdep for k in 1:s-1            # R_ks feeds y_k, conj(R_ks) feeds y_s
                        Rks = R[packed(k, s), f]
                        ys += conj(Rks) * p[k]
                        y[k] += Rks * ps
                    end
                    y[s] = ys + R[packed(s, s), f] * ps   # y_s is first set here
                end
                P̂[f, :] = y
            end
        end
        cols!(Z, inv, P̂)
    end
    view(P, 1:M, :), view(Z, 1:M, :), U!
end

# CG on U a = b. Returns a, the CG coefficients α and β, and the symbol R of U.
function design(G_ej, G_jr, h, pairs, c, M, iters)
    N, K, nE = size(G_ej)
    n = fftlen(M, N)
    Ge = spectra(G_ej, n)
    T = spectra(G_jr[:, :, first.(pairs) .+ 1], n) .* Ge[:, :, last.(pairs) .+ 1]  # routed pairs q
    S = gram(G_jr, nextprod((2, 3, 5), 2N - 1), n)
    R = Matrix{ComplexF64}(undef, packed(K, K), n ÷ 2 + 1)
    pfor(size(R, 2)) do fs              # R_ks = (c E_ks S_ks + (1 - c) sum_q conj(T_qk) T_qs) / n
        @inbounds for f in fs, s in 1:K, k in 1:s
            E = sum(conj(Ge[f, k, i]) * Ge[f, s, i] for i in 1:nE)
            Tks = sum(conj(T[f, k, q]) * T[f, s, q] for q in axes(T, 3))
            R[packed(k, s), f] = (c * E * S[f, packed(k, s)] + (1 - c) * Tks) / n
        end
    end
    r = irfft(sum(conj.(T) .* rfft([h; zeros(n - length(h))]); dims=3), n, 1)[1:M, :]  # b, as a = 0
    p, Up, U! = matvec(R, n, M, K)
    a, α, β, rs = zeros(M, K), zeros(iters), zeros(iters), coldot(r, r)
    p .= r
    for it in 1:iters
        U!()
        α[it] = rs / coldot(p, Up)
        rs⁺ = psum(K) do k
            @views a[:, k] .+= α[it] .* p[:, k]
            @views sum(abs2, r[:, k] .-= α[it] .* Up[:, k])
        end
        β[it], rs = rs⁺ / rs, rs⁺
        pcols(k -> @views(p[:, k] .= r[:, k] .+ β[it] .* p[:, k]), K)
    end
    a, α, β, R
end

# F p: spectra (n ÷ 2 + 1, n_E, n_R) of the responses of all pairs to the filter p, scaled by sw;
# F' u: first M samples of the adjoint. Returns p (M, K), F! and Ft!.
function lsq_operator(G_ej, G_jr, sw, M, n)
    Ge, Gj = spectra(G_ej, n), spectra(G_jr, n)
    (nf, K, nE), nR = size(Ge), size(Gj, 3)
    P, Z, Â, B = zeros(n, K), zeros(n, K), zeros(ComplexF64, nf, K), zeros(ComplexF64, nf, nE, K)
    Y, fwd, inv = zeros(ComplexF64, nf, nE, nR), plan_rfft(zeros(n)), iplan(n)
    function F!()
        cols!(Â, fwd, P)
        pfor(nf) do fs
            @inbounds for k in 1:K, i in 1:nE, f in fs
                B[f, i, k] = Ge[f, k, i] * Â[f, k]
            end
            @inbounds for j in 1:nR
                Y[fs, :, j] .= 0
                for k in 1:K, i in 1:nE
                    @simd for f in fs
                        Y[f, i, j] += B[f, i, k] * Gj[f, k, j]
                    end
                end
                @views Y[fs, :, j] .*= sw[:, j]'
            end
        end
        Y
    end
    function Ft!(u)
        pfor(nf) do fs
            B[fs, :, :] .= 0
            @inbounds for j in 1:nR, k in 1:K, i in 1:nE
                s = sw[i, j]
                @simd for f in fs
                    B[f, i, k] += conj(Gj[f, k, j]) * (s * u[f, i, j])
                end
            end
            @inbounds for k in 1:K, f in fs
                Â[f, k] = sum(conj(Ge[f, k, i]) * B[f, i, k] for i in 1:nE) / n
            end
        end
        view(cols!(Z, inv, Â), 1:M, :)
    end
    view(P, 1:M, :), F!, Ft!
end

# LSQR (Paige and Saunders) on F a = d with the residuals held as spectra, whose norm with the
# Parseval weights w is the time-domain one.
function design_lsqr(G_ej, G_jr, h, pairs, c, M, iters)
    (N, K, nE), nR = size(G_ej), size(G_jr, 3)
    n = fftlen(M, N)
    sw, d = fill(sqrt(c), nE, nR), zeros(ComplexF64, n ÷ 2 + 1, nE, nR)
    for (j, i) in pairs
        sw[i+1, j+1], d[:, i+1, j+1] = 1, rfft([h; zeros(n - length(h))])
    end
    v, F!, Ft! = lsq_operator(G_ej, G_jr, sw, M, n)
    u, w = zero(d), [1; fill(2.0, n ÷ 2 - 1); 1]
    function bidiag!(x, α)                  # β⁺ u = x - α u, α⁺ v = F'u - β⁺ v; returns β⁺, α⁺
        β⁺ = sqrt(psum(nR) do j
            acc = 0.0
            @inbounds for i in 1:nE
                @simd for f in eachindex(w)
                    u[f, i, j] = x[f, i, j] - α * u[f, i, j]
                    acc += w[f] * abs2(u[f, i, j])
                end
            end
            acc
        end / n)
        u ./= β⁺
        v .= Ft!(u) .- β⁺ .* v
        α⁺ = sqrt(coldot(v, v))
        v ./= α⁺
        β⁺, α⁺
    end
    β, α = bidiag!(d, 0.0)                                  # v = 0: β u = d, α v = F'u
    a, z, ϕ̄, ρ̄ = zeros(M, K), copy(v), β, α
    for _ in 1:iters
        β, α = bidiag!(F!(), α)
        ρ = hypot(ρ̄, β)
        cs, sn = ρ̄ / ρ, β / ρ
        θ, ρ̄ = sn * α, -cs * α
        ϕ, ϕ̄ = cs * ϕ̄, sn * ϕ̄
        @. a += (ϕ / ρ) * z
        @. z = v - (θ / ρ) * z
    end
    a
end
