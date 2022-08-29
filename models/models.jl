module Models
export DeepLabV3, PSPNet, UNet, get_model

using JSON
using Pipe: @pipe

include("unet.jl")
include("deeplab.jl")
include("pspnet.jl")

function get_model(model_name::String)
    models = Dict(
        "U-Net" => UNet(5, 2, [32, 64, 128, 256, 512, 1024]), 
        "DeepLabV3+" => Chain(Conv((1, 1), 5=>3, relu, pad=SamePad()), DeepLabV3(2, :ResNet34, true)), 
        "PSPNet" => Chain(Conv((1, 1), 5=>3, relu, pad=SamePad()), PSPNet(512, 2, :ResNet34, true)), 
    )
    return models[model_name] |> gpu
end

end