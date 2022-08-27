include("utils.jl")

function precision(ŷ::Array{Float32,4}, y::Array{Float32,4})
    ϵ = eps(Float32)
    ŷ = prediction_to_onehot(ŷ)
    tp = true_positives(ŷ, y)
    fp = false_positives(ŷ, y)
    return (tp .+ ϵ)./ (tp .+ fp .+ ϵ)
end

function precision(ŷ::Array{Float32,4}, y::Array{Float32,4}, agg::Function)
    return precision(ŷ, y) |> agg
end