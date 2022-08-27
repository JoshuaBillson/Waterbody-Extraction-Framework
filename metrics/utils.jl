using Pipe: @pipe
using Statistics: mean

function prediction_to_onehot(ŷ::Array{Float32,4})
    rows, cols, _, obs = size(ŷ)
    onehot = zeros(Float32, size(ŷ))
    for row in 1:rows, col in 1:cols, ob in 1:obs
        index = argmax(ŷ[row,col,:,ob])
        onehot[row,col,index,ob] = 1.0f0
    end
    return onehot
end

function true_positives(ŷ::Array{Float32,4}, y::Array{Float32,4})
    return @pipe ŷ .* y |> sum(_, dims=(1, 2, 4)) |> reshape(_, size(y)[3])
end

function false_positives(ŷ::Array{Float32,4}, y::Array{Float32,4})
    return @pipe ŷ .* (1 .- y) |> sum(_, dims=(1, 2, 4)) |> reshape(_, size(y)[3])
end

function false_negatives(ŷ::Array{Float32,4}, y::Array{Float32,4})
    return @pipe (1 .- ŷ) .* y |> sum(_, dims=(1, 2, 4)) |> reshape(_, size(y)[3])
end
