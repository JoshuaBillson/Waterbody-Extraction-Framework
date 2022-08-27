using CSV, DataFrames
using Pipe: @pipe

struct CSVCallback
    logdir::String
    metrics::Vector{Tuple{String,Function}}
end

function csv_callback(logdir::String, metrics::Vector{Tuple{String,Function}})
    mkpath(logdir)
    CSVCallback(logdir, metrics)
end

function (callback::CSVCallback)(model, data, epoch::Int)
    # Evaluate Performance On Test Set
    metrics = foldl(data, init=zeros(Float32, length(callback.metrics))) do acc, (x, y)
        ŷ, y = model(x) |> Array, Array(y)
        return acc .+ [metric(ŷ, y) for (_, metric) in callback.metrics]
    end

    # Construct DataFrame
    coltypes = [Int, [Float64 for i in 1:length(callback.metrics)]...]
    colnames = [:epoch, [Symbol(name) for (name, _) in callback.metrics]...]
    df = DataFrame(colnames .=> [type[] for type in coltypes])

    # Add Row To DataFrame
    names = [Symbol(name) for (name, _) in callback.metrics]
    d = Dict([Symbol(name)=>val for (name, val) in zip(names, metrics ./ length(data))]...)
    d[:epoch] = epoch
    push!(df, d)

    # Log Model Performance
    log_exists = "log.csv" in readdir(callback.logdir)
    @pipe joinpath(callback.logdir, "log.csv") |> CSV.write(_, df, append=log_exists)
end