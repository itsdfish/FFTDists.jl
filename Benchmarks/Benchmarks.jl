using DataFrames

function benchmarker(model, N)
    df = DataFrame(N=Int[], time=Float64[])
    for n in N
        data = rand(Uniform(0, 2), n)
        results = @benchmark run_model($model, $data)
        meanTime = mean(results).time * 1e-6
        temp = (N = n,time = meanTime)
        push!(df, temp)
    end
    return df
end

function run_model(model, points)
    convolve!(model)
    pdf.(model, points)
    return nothing
end
