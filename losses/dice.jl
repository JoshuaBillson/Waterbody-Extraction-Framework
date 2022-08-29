using Statistics

function dice_loss(ŷ::AbstractArray{Float32, 4}, y::AbstractArray{Float32, 4})
    ϵ = eps(Float32)
    intersection = sum(ŷ .* y, dims=(1, 2, 4))
    union = sum(ŷ, dims=(1, 2, 4)) .+ sum(y, dims=(1, 2, 4))
    dice_coefficient = ((2.0f0 .* intersection) .+ ϵ) ./ (union .+ ϵ)
    return mean(1.0f0 .- dice_coefficient)
end