using Flux, Metalhead

include("utils.jl")

function ASPP(backbone_name::Symbol)
    backbone_out_channels = backbone_name == :ResNet34 ? 512 : 2048
    out_pool = Chain(GlobalMeanPool(), ConvBlock(1, backbone_out_channels, 256, 1), Upsample(:bilinear, size=(16, 16)))
    out_1 = ConvBlock(1, backbone_out_channels, 256, 1)
    out_6 = ConvBlock(3, backbone_out_channels, 256, 6)
    out_12 = ConvBlock(3, backbone_out_channels, 256, 12)
    out_18 = ConvBlock(3, backbone_out_channels, 256, 18)
    return Chain(Parallel((x...) -> cat(x..., dims=3), out_pool, out_1, out_6, out_12, out_18), ConvBlock(1, 1280, 256, 1))
end

function DeepLabV3(nclasses::Int, backbone_name::Symbol, pretrain::Bool=false)
    # Get Backbone
    backbone = get_backbone(backbone_name, pretrain)

    # Get Backbone Outputs
    path1 = get_layer(backbone, "conv2_3", backbone_name)
    path2 = get_layer(backbone, "conv5_3", backbone_name)

    # Construction ASPP
    path_a = Chain(path2, ASPP(backbone_name), Upsample(:bilinear, size=(128, 128)))
    path_b = Chain(path1, ConvBlock(1, backbone_name == :ResNet34 ? 64 : 256, 48))

    # Build DeepLabV3
    Chain(
        Parallel((x...) -> cat(x..., dims=3), path_a, path_b), 
        ConvBlock(3, 304, 256), 
        ConvBlock(3, 256, 256), 
        Upsample(:bilinear, scale=(4, 4)), 
        Conv((3, 3), 256=>nclasses, dilation=1, pad=SamePad()),
        x -> softmax(x, dims=3))
end