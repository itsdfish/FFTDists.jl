using Base.MathConstants

"""
cf is a closed form approximation for the Lognormal characteristic function.
See the following for details: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.720.1797&rep=rep1&type=pdf
"""
function cf(d::LogNormal, θ)
    θ = -θ*im; σ = d.σ; μ = d.μ
    a = lambertw(θ*e^μ*σ^2)^2
    b = 2*lambertw(θ*e^μ*σ^2)
    d = sqrt(1+lambertw(θ*e^μ*σ^2))
    return exp(-(a + b)/(2*σ^2))/d
end


mutable struct Power <: ContinuousUnivariateDistribution
    L::Float64
    U::Float64
    d::Float64
end

function cf(dist::Power, t)
    L = dist.L; U = dist.U; d=dist.d
    Γ = inc_gamma_lower
    x1 = (-t+0im)^(1/d)*exp(1im*π/(2*d))*gamma(-1/d)*Γ(exp(1im*π/d),L^(-d)*t*exp(1im*π/2))/(d*gamma(1-1/d))
    x2 = (-t+0im)^(1/d)*exp(1im*π/(2*d))*gamma(-1/d)*Γ(exp(1im*π/d),U^(-d)*t*exp(1im*π/2))/(d*gamma(1-1/d))
    return x1 + x2
end
