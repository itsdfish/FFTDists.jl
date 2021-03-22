__precompile__()
module FFTDists
    using Distributions, Dierckx, StatsBase
    using FFTW, LambertW, Statistics
    import Base: +
    import Distributions: pdf, logpdf, cf, rand
    export convolve!, +, pdf, logpdf, Extras, rand
    export Convolution, simulate, numeric, convolve_normal
    include("FFTfunctions.jl")
    include("Utilities.jl")
end
