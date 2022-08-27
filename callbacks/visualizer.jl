include("../pipelines/pipelines.jl")
using .Pipelines

include("utils.jl")

struct VisualizerCallback
    samples::Vector{Int}
    folder::String
end

function visualizer_callback(samples::Vector{Int}, folder::String)
    @assert length(samples) == 7 "Error: Must Provide 7 Samples! (Received $(length(samples)))"
    mkpath(folder)
    VisualizerCallback(samples, folder)
end

function (callback::VisualizerCallback)(model, data, epoch)
    figfile = joinpath(callback.folder, "prediction_$epoch.png")
    @pipe [show_prediction(model, tile) for tile in callback.samples] |> showimg(_, (5000, 7000), (7, 1)) |> savefig(_, figfile)
end

"""
Take a prediction tensor produced by pixel-wise softmax and transform it into a sparsely labelled mask.

`ŷ`: The prediction to be cast into a mask.
"""
function prediction_to_mask(ŷ::Array{Float32,4})
	(mapslices(argmax, ŷ, dims=3) .|> Float32) .- 1.0f0
end

"""
Make and plot the prediction produced by the given model when passed the features belonnging to the given tile.  
The resulting plot will consist of 5 sub-plots in a single row in the order of RGB, NIR, SWIR, Mask, Prediction.

`model`: The model which will make the prediction.  
`tile`: The tile for which we want to make the prediction.  
"""
function show_prediction(model, tile::Int)
	# Plot Features
	rgb_plot = read_rgb(tile) |> plot_color
	nir_plot = read_nir(tile) |> plot_gray
	swir_plot = read_swir(tile) |> plot_gray
	mask_plot = read_mask(tile) |> plot_gray
	
	# Plot Prediction
	x, _ = Pipelines.ImagePipeline([tile])[1]
    prediction_plot = @pipe model(x) |> Array |> prediction_to_mask(_)[:,:,:,1] |> plot_gray

	# Plot Features And Prediction
	showimg([rgb_plot, nir_plot, swir_plot, mask_plot, prediction_plot], (5000, 1000), (1, 5))
end
