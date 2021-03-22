"""
The Dist object contains information for convolving random variables.
* `lb`: lower bound of density
* `ub`: upper bound of density
* `n_points`: number of discrete points used in transform
* `x`: real values associated with densities
* `density`: densities resulting from Convolution
* `interp_pdf`: an object used to interpolate exact densities from x and density
```@julia
model = Normal(0,1) + Normal(0,1)
convolve!(model)

julia> pdf(model, 0.0)
0.28216441829987216

julia> pdf(Normal(0,sqrt(2)), 0.0)
0.28209479177387814
```
"""
mutable struct Convolution{T}
    model::T
    lb::Float64
    ub::Float64
    n_points::Int64
    x::Vector{Float64}
    density::Vector{Float64}
    interp_pdf::Spline1D
end

function Convolution(;model, lb=-3.0, ub=3.0, n_points = 2^9, x=fill(0.0, 0),
        density=fill(0.0, 0), interp_pdf=Spline1D(zeros(4), zeros(4)))
    Convolution(model, lb, ub, n_points, x, density, interp_pdf)
end

Broadcast.broadcastable(c::Convolution) = Ref(c)

function +(d::Distribution...)
    convolution = Convolution(model=d)
    return convolution
end

function analytic(D::Convolution)
    N = D.n_points;  model = D.model
    lb = D.lb;  ub = D.ub
    i = range(0, length=N); dx = (ub-lb)/N
    x = lb .+ i*dx; dt = 2*π/(N*dx)
    c = -N/2*dt; t = c .+ i*dt; f = fill(1im+1.0, N)
    for m in model
        @. f = f*cf(m, t)
    end
    X = @. exp(-(0+1im)*i*dt*lb)*f
    fft!(X)
    density = @. dt/(2*π)*exp(-(0+1im)*c*x)*X
    # Use abs()to haldle very small negative values, e.g. -2.50e-12
    D.density = abs.(real(density))
    D.x = x
    D.interp_pdf = Spline1D(D.x, D.density)
    return nothing
end

function numeric(D::Convolution)
   n_points = D.n_points;  model = D.model
   lb = D.lb;  ub = D.ub
   Ndist = length(model)
   stepsize = (ub-lb)/n_points
   x = range(lb, stop=ub, length=n_points)
   d = pdf.(model[1], x)
   mn = Int(round(1.5*n_points)) + 1
   st = Int(round(2*n_points))
   mx = n_points ÷ 2
   v = [fill(0.0, mx); d ;fill(0.0, mx)]
   Nv = length(v)
   d1 = pdf.(model[2], x)
   v1 = [fill(0.0, mx); d1; fill(0.0, mx)]
   m = fft(v).*fft(v1)
   cv = real(ifft(m))*Nv*stepsize/(2*n_points)
   cv = abs.(cv[[mn:st;1:mx]])
   cnt = 2
   while cnt < Ndist
       cnt += 1
       v = [fill(0.0, mx); cv; fill(0.0, mx)]
       d1 = pdf.(model[cnt], x)
       v1 = [fill(0.0, mx); d1 ;fill(0.0, mx)]
       m = fft(v).*fft(v1)
       cv = real(ifft(m))*Nv*stepsize/(2*n_points)
       cv = abs.(cv[[mn:st;1:mx]])
    end
    D.density = cv
    D.x = x
    D.interp_pdf = Spline1D(D.x, D.density)
    return nothing
end

"""
Convolves a set of random variables.
```@julia
model = Normal(0,1) + Normal(0,1)
convolve!(model)

julia> pdf(model, 0.0)
0.28216441829987216

julia> pdf(Normal(0,sqrt(2)), 0.0)
0.28209479177387814
```
"""
function convolve!(D::Convolution)
    if all(x->applicable(cf, (x,1.0)...), D.model)
        # Convolve with characteristic function if possible
        analytic(D)
    else
        numeric(D)
    end
    return nothing
end

 function rand(model::Convolution, N::Int64)
     rts = fill(0.0, N)
     dist = model.model
     for d in dist
         rts .+= rand(d, N)
     end
     return rts
 end

Gradient(d::Convolution, x::Float64) = d.interpGrad[x]
pdf(d::Convolution, x::Float64) = abs(d.interp_pdf(x))
logpdf(d::Convolution, x::Float64) = log(pdf(d, x))

"""
Normal approximation to the sum of Uniform rvs
* 'scaling' = (2/3): default scaling used for standard deviation σ
* 'args': a list of NamedTuples for component distributions, e.g. cr =(μ=.05,N=2),...

```@julia

julia> μ,σ = convolve_normal(cr=(μ=.05,N=2),vis=(μ=.085,N=1))
(0.185, 0.021278575558006173)
```
"""
function convolve_normal(;scaling=2/3, args...)
    # uniform factor
    fact1 = 1/12
    μ,σ = 0.0,0.0
    for (k,v) in pairs(args)
        μ += prod(v)
        σ += fact1.*v.N*(scaling*v.μ)^2
    end
    σ = sqrt(σ)
    return μ,σ
end
