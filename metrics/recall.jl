include("utils.jl")

function recall(ŷ::Array{Float32,4}, y::Array{Float32,4})
    ϵ = eps(Float32)
    ŷ = prediction_to_onehot(ŷ)
    tp = true_positives(ŷ, y)
    fn = false_negatives(ŷ, y)
    return (tp .+ ϵ) ./ (tp .+ fn .+ ϵ)
end

function recall(ŷ::Array{Float32,4}, y::Array{Float32,4}, agg::Function)
    return recall(ŷ, y) |> agg
end