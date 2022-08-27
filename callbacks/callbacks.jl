module Callbacks
export CSVCallback
export VisualizerCallback, CSVCallback, visualizer_callback, csv_callback, CallbackManager, CheckpointCallback, checkpoint_callback

include("csv_logger.jl")
include("visualizer.jl")
include("checkpoint_logger.jl")

struct CallbackManager
    callbacks
end

function(cm::CallbackManager)(model, data, epoch::Int)
    for callback in cm.callbacks
        callback(model, data, epoch)
    end
end

end