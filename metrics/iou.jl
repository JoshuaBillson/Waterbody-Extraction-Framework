include("utils.jl")

function IoU(ŷ::Array{Float32,4}, y::Array{Float32,4})
    ϵ = eps(Float32)
    ŷ = prediction_to_onehot(ŷ)
    tp = true_positives(ŷ, y)
    fn = false_negatives(ŷ, y)
    fp = false_positives(ŷ, y)
    return (tp .+ ϵ) ./ (tp .+ fp .+ fn .+ ϵ)
end

function mIoU(ŷ::Array{Float32,4}, y::Array{Float32,4})
    return IoU(ŷ, y) |> mean
end