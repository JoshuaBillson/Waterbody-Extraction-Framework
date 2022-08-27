using Flux
using Pipe: @pipe

function get_layer_index(blocks::AbstractArray{Int}, layer_name::String)
    @assert length(blocks) == 4
    conv, block = @pipe match(r"^conv(\d)_(\d)$", layer_name) .|> parse(Int, _)
    @assert 2 <= conv <= 5
    @assert 1 <= block <= blocks[conv-1]
    return sum(blocks[1:conv-2]) + 3 + block
end

function get_layer(model, layer_name::String, model_type::Symbol)
    blocks = Dict(
        :ResNet34 => [3, 4, 6, 3], 
        :ResNet50 => [3, 4, 6, 3], 
        :ResNet101 => [3, 4, 23, 3], 
        :ResNet152 => [3, 8, 36, 3])
    
    @assert haskey(blocks, model_type) "Error: Invalid Model Type Given!"
    return @pipe blocks[model_type] |> get_layer_index(_, layer_name) |> model.layers[1][1:_] |> Chain(_...)
end

function get_layer(model, start_layer::String, end_layer::String, model_type::Symbol)
    blocks = Dict(
        :ResNet34 => [3, 4, 6, 3], 
        :ResNet50 => [3, 4, 6, 3], 
        :ResNet101 => [3, 4, 23, 3], 
        :ResNet152 => [3, 8, 36, 3])
    
    @assert haskey(blocks, model_type) "Error: Invalid Model Type Given!"
    start_index = @pipe blocks[model_type] |> get_layer_index(_, start_layer)
    end_index = @pipe blocks[model_type] |> get_layer_index(_, end_layer)
    return Chain(model.layers[1][start_index:end_index]...)
end

function get_backbone(backbone::Symbol, pretrain::Bool=false)
    backbones = Dict(
        :ResNet34 => () -> ResNet(34, pretrain=pretrain), 
        :ResNet50 => () -> ResNet(50, pretrain=pretrain), 
        :ResNet101 => () -> ResNet(101, pretrain=pretrain), 
        :ResNet152 => () -> ResNet(152, pretrain=pretrain) )
    return backbones[backbone]()
end

function ConvBlock(kernel::Int, in_filters::Int, out_filters::Int, dilation::Int=1)
    Chain( Conv((kernel, kernel), in_filters=>out_filters, pad=SamePad(), dilation=dilation), BatchNorm(out_filters, relu) )
end