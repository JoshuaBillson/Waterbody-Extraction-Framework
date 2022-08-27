using Plots, Images, ArchGDAL, Flux, MLUtils, CUDA, Random, ProgressBars, Statistics, JSON
using Pipe: @pipe

include("models/models.jl")
using .Models

include("losses/losses.jl")
using .Losses

include("pipelines/pipelines.jl")
using .Pipelines

include("callbacks/callbacks.jl")
using .Callbacks

include("metrics/metrics.jl")
using .Metrics

const SAMPLE_TILES = [114, 236, 318, 669, 676, 790, 991]

function load_data(at::AbstractFloat, train_batch_size::Int, test_batch_size::Int)
	# List All Tiles
    Random.seed!(12)
    tiles = collect(1:1600)
    filter!(x -> !(x in SAMPLE_TILES), tiles)
    shuffle!(tiles)

	# Split Tiles Into Training And Test
	split_index = 1600 * at |> floor |> Int
	train = Pipelines.ImagePipeline(tiles[1:split_index])
	test = Pipelines.ImagePipeline(tiles[split_index+1:end])

	# Construct DataLoaders
	train_data = DataLoader(train, batchsize=train_batch_size, shuffle=true)
	test_data = DataLoader(test, batchsize=test_batch_size, shuffle=false)
	return train_data, test_data
end

function train_model(model, train_data, validation_data, opt, loss, epochs::Int, callbacks)
    # Iterate Over Epochs
	params = Flux.params(model)
	for epoch in 1:epochs

		# Iterate Over Mini-Batches
        l = 0.0f0
        iter = ProgressBar(train_data)
		for (i, (x, y)) in enumerate(iter)

            water_content = @pipe sum(y[:,:,2,:]) |> /(_, 512 * 512) |> *(_, 100)

            if water_content > 0.0

                # Compute Gradients
                grads = Flux.gradient(() -> loss(x, y), params)

                # Update Parameters
                Flux.Optimise.update!(opt, params, grads)

                current_loss = loss(x, y)
                l += current_loss
                set_description(iter, "Loss: $(round(current_loss, digits=4, base=10)), Average Loss: $(round(l / Float32(i), digits=4, base=10))")
            end
		end

		# Run Callbacks At End Of Epoch
		callbacks(model, validation_data, epoch)

	end
end

function main()
	# Load Config File
	config = joinpath(pwd(), "config.json") |> JSON.parsefile

	# Load Training Data
	train_data, test_data = load_data(0.8, config["batch_size"], 8)

	# Fetch Model
	model = Models.get_model(config["model"])

	# Create Metrics
	miou(ŷ, y) = Metrics.mIoU(ŷ, y)
	recall(ŷ, y) = Metrics.recall(ŷ, y)[2]
	precision(ŷ, y) = Metrics.precision(ŷ, y)[2]
	metrics = [("mIoU", miou), ("recall", recall), ("precision", precision)]

	# Create Callbacks
	csv_logger = @pipe joinpath(pwd(), "experiments", config["experiment"], "logs") |> Callbacks.csv_callback(_, metrics)
	visualizer = @pipe joinpath(pwd(), "experiments", config["experiment"], "visualizations") |> Callbacks.visualizer_callback(SAMPLE_TILES, _)
	checkpoint_logger = joinpath(pwd(), "experiments", config["experiment"], "checkpoints") |> Callbacks.checkpoint_callback
	callbacks = Callbacks.CallbackManager([csv_logger, visualizer, checkpoint_logger])

	# Define Optimizer
	η = config["learning_rate"]
	opt = Flux.Optimiser(ClipNorm(η), Adam(η))

    # Define Loss
	loss = Losses.get_loss(model, config["loss"])

	# Train The Model
	train_model(model, train_data, test_data, opt, loss, config["epochs"], callbacks)

	return model
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end