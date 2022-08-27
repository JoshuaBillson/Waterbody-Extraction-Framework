using BSON: @save
using Flux

struct CheckpointCallback
    checkpoint_directory::String
end

function checkpoint_callback(checkpoint_directory::String)
    mkpath(checkpoint_directory)
    CheckpointCallback(checkpoint_directory)
end

function (callback::CheckpointCallback)(model, data, epoch::Int)
    filename = joinpath(callback.checkpoint_directory, "checkpoint_$epoch.bson")
    model = model |> cpu
    @save filename model
end