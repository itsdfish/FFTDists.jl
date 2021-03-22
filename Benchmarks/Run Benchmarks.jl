using Revise, BenchmarkTools, DataFrames
using Distributions, StatsPlots, FFTDists, Plots.Measures
include("Benchmarks.jl")
############################################################
#####################  Define Models  ######################
############################################################
model = Gamma(5, .05) + LogNormal(-1, .5)
N = 10 .^(0:7)
############################################################
#####################  Benchmark  Model  ###################
############################################################
results = benchmarker(model, N)
############################################################
#####################  Plot Model  #########################
############################################################
pyplot()
options = (norm=true, legend=false, ylabel="Mean Time (ms)", xlabel="N")
@df results plot(:N, :time, xaxis=:log; options...)
