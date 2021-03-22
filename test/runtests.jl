using SafeTestsets

@safetestset "FFT Tests" begin
    using FFTDists, StatsBase, Statistics, Distributions, Test, Random
    import StatsBase: normalize
    
    function trueXY(edges, data)
        h = fit(Histogram, data, edges)
        h = normalize(h)
        y = h.weights
        Δ = step(edges)*.5
        x = edges[1:end-1] .+ Δ
        return x,y
    end
    
    meanAbs(v1, v2) = mean(abs.(v1-v2))
    ############################################################
    #####################  Define Model  #######################
    ############################################################
    Random.seed!(45484)
    model = LogNormal(-1,.7) + LogNormal(-1,.7)
    convolve!(model)
    edges = 0:.01:2
    trueVals = rand(model,10^6)
    x,y = trueXY(edges, trueVals)
    dens = pdf.(model, x)
    @test meanAbs(y, dens) ≈ 0 atol= .05
    ############################################################
    #####################  Define Model  #######################
    ############################################################
    Random.seed!(4544)
    model = Gamma(5,.05) + LogNormal(-1,.7)
    convolve!(model)
    edges = 0:.01:2
    trueVals = rand(model, 10^6)
    x,y = trueXY(edges, trueVals)
    dens = pdf.(model, x)
    @test meanAbs(y, dens) ≈ 0 atol= .05
    ############################################################
    #####################  Define Model  #######################
    ############################################################
    Random.seed!(4547)
    model = Uniform(0,1) + InverseGaussian(.5,.5)
    convolve!(model)
    edges = 0:.05:2
    trueVals = rand(model, 10^6)
    x,y = trueXY(edges, trueVals)
    dens = pdf.(model, x)
    @test meanAbs(y, dens) ≈ 0 atol= .05
    ############################################################
    #####################  Define Model  #######################
    ############################################################
    Random.seed!(555)
    model = Uniform(0,.5) + Uniform(0,.5)
    convolve!(model)
    edges = 0:.01:2
    trueVals = rand(model, 10^6)
    x,y = trueXY(edges, trueVals)
    dens = pdf.(model, x)
    @test meanAbs(y, dens) ≈ 0 atol= .05
    ############################################################
    #####################  Define Model  #######################
    ############################################################
    Random.seed!(985)
    model = Uniform(.5,1.5) + Uniform(0,.1^6)
    convolve!(model)
    edges = 0:.01:2
    trueVals = rand(model, 10^6)
    x,y = trueXY(edges, trueVals)
    dens = pdf.(model, x)
    @test meanAbs(y, dens) ≈ 0 atol= .05
    ############################################################
    #####################  Define Model  #######################
    ############################################################
    μ,σ = convolve_normal(motor=(μ=.21,N=1), cr=(μ=.05,N=2))
    x = rand(Uniform(.21 - .21/3, .21 + .21/3), 10^5) + rand(Uniform(.05 - .05/3, .05 + .05/3), 10^5) + 
        rand(Uniform(.05 - .05/3, .05 + .05/3), 10^5)
    @test μ ≈ .21 + .05*2
    @test σ ≈ sqrt((.21/3*sqrt(1/3))^2 + 2*(.05/3*sqrt(1/3))^2) 
    @test σ ≈ std(x) rtol = .01
end