using Flux, Metalhead

include("utils.jl")

function PyramidPoolingLayer(scale::Int, input_channels::Int=256)
    @assert input_channels / 4 |> isinteger "Error: Input Channels Must Be Divisible By 4!"
    Chain( MeanPool((scale, scale)), ConvBlock(1, input_channels, (input_channels ÷ 4)), Upsample(scale, :bilinear) )
end

function GlobalPoolingLayer(sz::Int, input_channels::Int=256)
    Chain( GlobalMeanPool(), ConvBlock(1, input_channels, (input_channels ÷ 4)), Upsample(:bilinear, size=(sz, sz)) )
end

function PyramidPoolingModule(input_size::Int, input_channels::Int=256)
    Parallel( 
        (x...) -> cat(x..., dims=3), 
        GlobalPoolingLayer(input_size ÷ 16, input_channels), 
        PyramidPoolingLayer(2, input_channels), 
        PyramidPoolingLayer(4, input_channels), 
        PyramidPoolingLayer(8, input_channels), 
        identity
     )
end

function PSPNet(input_size::Int, nclasses::Int, backbone_name::Symbol, pretrain::Bool=false)
    backbone_out_layers = Dict(
        :ResNet34 => "conv4_6", 
        :ResNet50 => "conv4_6", 
        :ResNet101 => "conv4_23", 
        :ResNet152 => "conv4_36" )

    # Build Backbone
    backbone = get_backbone(backbone_name, pretrain)

    # Build Pyramid Pooling Module
    backbone_out_channels = backbone_name == :ResNet34 ? 256 : 1024
    ppm = PyramidPoolingModule(input_size, backbone_out_channels)

    # Build Activation Layer
    activation = Chain( 
        Upsample(16, :bilinear), 
        Conv((3, 3), (backbone_out_channels * 2)=>128, relu, pad=SamePad()), 
        Conv((1, 1), 128=>nclasses, pad=SamePad()), 
        x -> softmax(x, dims=3) )

    # Build PSPNet
    Chain( get_layer(backbone, backbone_out_layers[backbone_name], backbone_name), ppm, activation )
end
