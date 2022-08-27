module Losses
export dice_loss, get_loss

using JSON
using Pipe: @pipe

include("dice.jl")

function get_loss(model, loss)
    losses = Dict(
        "dice" => (x, y) -> dice_loss(model(x), y) )
    return losses[loss]
end

end