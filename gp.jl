using PGFPlots
using Distributions
using Optim
using Parameters

import LinearAlgebra: norm, I, diag

const Dset = Vector{Vector{Float64}}

function myopt(f; a=0.0, b=8.0, Δx=0.1)
    best_x = a
    best_y = Inf
    m = ceil(Int, (b-a)/Δx)
    pts = range(a, stop=b, length=m+1)
    for i in 1 : m
        res = optimize(f, pts[i], pts[i+1])
        if res.minimum < best_y
            best_y = res.minimum
            best_x = res.minimizer
        end
    end
    return best_x
end

μ(X::Dset, m::Function) = [m(x) for x in X]
Σ(X::Dset, k::Function) = [k(x,x′) for x in X, x′ in X]
K(X::Dset, X′::Dset, k::Function) = [k(x,x′) for x in X, x′ in X′]

@with_kw struct GaussianProcess
    m::Function = x -> 0.0
    k::Function = (x,x′)->exp(-(norm(x-x′))^2)
    X::Vector{Vector{Float64}} = Vector{Float64}[]
    y::Vector{Float64} = Float64[]
    ν::Float64 = 0.0 # variance when sampling f
end

mvnrand(μ::Vector{Float64}, Σ::Matrix{Float64}, inflation=1e-6) = rand(MvNormal(μ, Σ + inflation*I));
Base.rand(GP::GaussianProcess, X::Dset) = mvnrand(μ(X, GP.m), Σ(X, GP.k))
function Base.push!(GP::GaussianProcess, x::Vector{Float64}, y::Real)
    push!(GP.X, x)
    push!(GP.y, y)
    return GP
end
function Base.pop!(GP::GaussianProcess)
    pop!(GP.X)
    pop!(GP.y)
    return GP
end

function predict(GP::GaussianProcess, X_pred::Dset)
    m, k, ν = GP.m, GP.k, GP.ν
    tmp = K(X_pred, GP.X, k) / (K(GP.X, GP.X, k) + ν*I)
    μₚ = μ(X_pred, m) + tmp*(GP.y - μ(GP.X, m))
    S = K(X_pred, X_pred, k) - tmp*K(GP.X, X_pred, k)
    νₚ = diag(S) .+ eps() # eps prevents numerical issues
    return (μₚ, νₚ)
end
function predict(GP::GaussianProcess, x_pred::Vector{Float64})
    (μₚ, νₚ) = predict(GP, [x_pred])
    return Normal(μₚ[1], sqrt(νₚ[1]))
end

prob_of_improvement(N::Normal{Float64}, y_max::Real) = isapprox(N.σ, 0, atol=1e-4) ? 0.0 : cdf(N, y_max)
prob_is_safe(N::Normal{Float64}, y_max::Real) = cdf(N, y_max)

function upperbound(GP::GaussianProcess, x_pred::Vector{Float64}, β::Real)
    (μₚ, νₚ) = predict(GP, [x_pred])
    return μₚ[1] + sqrt(β*νₚ[1])
end
function lowerbound(GP::GaussianProcess, x_pred::Vector{Float64}, β::Real)
    (μₚ, νₚ) = predict(GP, [x_pred])
    return μₚ[1] - sqrt(β*νₚ[1])
end
function width(GP::GaussianProcess, x_pred::Vector{Float64}, β::Real)
    (μₚ, νₚ) = predict(GP, [x_pred])
    return 2sqrt(β*νₚ[1])
end

is_safe(GP::GaussianProcess, x::Vector{Float64}, β::Real, y_max::Real) = upperbound(GP, x, β) <= y_max
function get_safe_regions(GP::GaussianProcess, β::Real, a::Real, b::Real, y_max::Real; Δx=(b-a)/201)

    a = 1.0*a
    b = 1.0*b

    x = Float64[a]
    safe_regions = Tuple{Float64,Float64}[]
    while x[1] ≤ b
        while !is_safe(GP, x, β, y_max) && x[1] < b
            x[1] += Δx
        end
        lo = x[1]
        if is_safe(GP, x, β, y_max)
            while is_safe(GP, x, β, y_max) && x[1] < b
                x[1] += Δx
            end
            hi = x[1]
            push!(safe_regions, (lo,hi))
            x[1] += Δx
        end
    end
    return safe_regions
end

function get_best_upperbound(GP::GaussianProcess, β::Real, safe_regions::Vector{Tuple{Float64,Float64}})
    best_hi = Inf
    for (a,b) in safe_regions
        best_hi = min(best_hi, myopt(x->upperbound(GP, [x], β), a=a, b=b))
    end
    return upperbound(GP, [best_hi], β)
end

function get_potential_maximizers(GP::GaussianProcess, β::Real, safe_regions::Vector{Tuple{Float64,Float64}}; Δx=0.01)

    best_hi = get_best_upperbound(GP, β, safe_regions)

    M_regions = Tuple{Float64,Float64}[]
    for (a,b) in safe_regions
        x = [a*1.0]
        while x[1] ≤ b
            while lowerbound(GP, x, β) > best_hi && x[1] < b
                x[1] += Δx
            end
            lo = x[1]
            if lowerbound(GP, x, β) ≤ best_hi
                while lowerbound(GP, x, β) ≤ best_hi && x[1] < b
                    x[1] += Δx
                end
                hi = x[1]
                push!(M_regions, (lo,hi))
                x .+= Δx
            else
                x .+= Δx
            end
        end
    end
    return M_regions
end

function is_expander(
    GP::GaussianProcess,
    x::Vector{Float64},
    β::Float64,
    a::Float64,
    b::Float64,
    L::Float64,
    d::Function,
    y_max::Float64,
    )

    ℓ = lowerbound(GP, x, β)
    return ℓ + L*min(d(x, [a]), d(x, [b])) ≤ y_max
end
function get_potential_expanders(
    GP::GaussianProcess,
    β::Real,
    safe_regions::Vector{Tuple{Float64,Float64}},
    L::Float64,
    d::Function,
    y_max::Float64,
    ;
    Δx = 0.01,
    )

    E_regions = Tuple{Float64,Float64}[]
    for (a,b) in safe_regions
        x = [a*1.0]
        while is_expander(GP, x, β, a, b, L, d, y_max) && x[1] < b
            x[1] += Δx
        end
        if !is_expander(GP, x, β, a, b, L, d, y_max) && x[1] > a
            push!(E_regions, (a, x[1]))
        end

        x = [b*1.0]
        while is_expander(GP, x, β, a, b, L, d, y_max) && x[1] > a
            x[1] -= Δx
        end
        if !is_expander(GP, x, β, a, b, L, d, y_max) && x[1] > a
            push!(E_regions, (x[1], b))
        end

    end

    return E_regions
end

function plot_GP_data(GP::GaussianProcess; legendentry::String="")
    p = Plots.Scatter([x[1] for x in GP.X], GP.y, style="only marks, mark=*, mark size=1, mark options={draw=black, fill=black}")
    if !isempty(legendentry)
        p.legendentry = legendentry
    end
    return p
end

function plot_transparent_interval(xdom::Tuple{A,B}, ydom::Tuple{C,D}; style::String="") where {A<:Real, B<:Real, C<:Real, D<:Real}
    a,b = xdom
    lo,hi = ydom

    p = Plots.Plot[]
    push!(p, Plots.Linear([a,a], [lo,hi], style="name path=A, draw=none, mark=none"))
    push!(p, Plots.Linear([b,b], [lo,hi], style="name path=B, draw=none, mark=none"))
    push!(p, Plots.Command("\\addplot[$style] fill between[of=A and B]"))
    return p
end

function plot_transparent_intervals(regions::Vector{Tuple{Float64,Float64}}, ydom::Tuple{Float64,Float64}, color::String, opacity::Real, legendentry::String="")
    p = Plots.Plot[]
    lo, hi = ydom
    for (i,tup) in enumerate(regions)
        a, b = tup

        push!(p, Plots.Linear([a,a], [lo,hi], style="name path=A, draw=none, mark=none, forget plot"))
        push!(p, Plots.Linear([b,b], [lo,hi], style="name path=B, draw=none, mark=none, forget plot"))

        if i != length(regions) || isempty(legendentry)
            push!(p, Plots.Command("\\addplot[$color, forget plot, opacity=$(string(opacity))] fill between[of=A and B]"))
        else
            push!(p, Plots.Command("\\addplot[$color, opacity=$(string(opacity))] fill between[of=A and B]"))
            push!(p, Plots.Command("\\addlegendentry{$legendentry}"))
        end
    end
    return p
end

function update_confidence_intervals!(GP, X, u, ℓ, β)
    μₚ, νₚ = predict(GP, X)
    u[:] = μₚ + sqrt.(β*νₚ)
    ℓ[:] = μₚ - sqrt.(β*νₚ)
    return (u, ℓ)
end
function compute_sets!(GP, S, M, E, X, u, l, y_max, β)
	fill!(M, false)
    fill!(E, false)

    # safe set
    S[:] = u .≤ y_max

    if any(S)

        # potential minimizers
        M[S] = l[S] .< minimum(u[S])

        # maximum width (in M)
        w_max = maximum(u[M] - l[M])

        # expanders - skip values in M or those with w ≤ w_max
        E[:] = S .& .~M # skip points in M
        if any(E)
            E[E] .= maximum(u[E] - l[E]) .> w_max
            for (i,e) in enumerate(E)
                if e && u[i] - l[i] > w_max
                    push!(GP, X[i], l[i])
                    μₚ, νₚ = predict(GP, X[.~S])
                    pop!(GP)
                    E[i] = any(μₚ + sqrt.(β*νₚ) .≥ y_max)
                    if E[i]; w_max = u[i] - l[i]; end
                end
            end
        end
    end

    return (S,M,E)
end
function compute_sets!(S, M, E, X, u, ℓ, y_max)
    # update the safe set based on current confidence bounds
    S[:] = u .≤ y_max

    # maximizers
    fill!(M, false)
    M[S] = u[S] .≥ maximum(ℓ[S])

    # maximum width (in M)
    w_max = maximum(u[M] - ℓ[M])

    # expanders
    #=
    For the run of the algorithm we do not need to calculate the
    full set of potential expanders:
    We can skip the ones already in M and ones that have lower
    variance than the maximum variance in M, w_max or the threshold.
    Amongst the remaining ones we only need to find the
    potential expander with maximum variance
    =#
    E[:] = S .& .~M # skip points in M
    if any(E)
        E[E] = maximum(u[E] - ℓ[E]) .> w_max # skip points with low width
        for (i,e) in enumerate(E)
            if e && u[i] - ℓ[i] > w_max # is potentially an expander and higher width
                push!(GP, X[i], ℓ[i]) # Add safe point with its lowest possible value to the GP
                μₚ, νₚ = predict(GP, X[.~S]) # Prediction of previously unsafe points based on that
                pop!(GP) # Remove the fake data point from the GP again
                E[i] = any(μₚ + sqrt.(β*νₚ) ≥ y_max) # If any unsafe upperr bound is suddenly below fmax then the point is an expander
                if E[i]; w_max = u[i] - ℓ[i]; end # so we don't consider other expanders with lower width
            end
        end
    end

    return (S,M,E)
end
function get_new_query_point(M, E, u, ℓ)
    ME = M .| E
    any(ME) || error("There are no points to evaluate")
    return something(findfirst(isequal(argmax(u[ME] - ℓ[ME])), cumsum(ME)), 0)
end

function safe_opt(GP, X, xi, f, y_max; β=3.0, K=10)
    push!(GP, X[i], f(X[i])) # make first observation

    m = length(X)
    u, ℓ = fill(Inf, m), fill(-Inf, m)
    S, M, E = falses(m), falses(m), falses(m)

    for k in 1 : K
        update_confidence_intervals!(GP, X, u, ℓ, β)
        compute_sets!(S, M, E, X, u, ℓ, y_max)
        i = get_new_query_point(M, E, u, ℓ)
        push!(GP, X[i], f(X[i]))
    end

    # return the best point
    update_confidence_intervals!(GP, X, u, ℓ, β)
    S[:] = u .≤ y_max
    if any(S)
        u_best, i_best = findmin(u[S])
        i_best = something(findfirst(isequal(i_best), cumsum(S)), 0)
        return (u_best, i_best)
    else
        return (NaN,0)
    end
end