using Flux, DataStructures
using Pipe: @pipe
include("utils.jl")

struct EncoderBlock
    conv_block
    downsample
end

Flux.@functor EncoderBlock

function EncoderBlock(in::Int, out::Int)
    conv_block = Chain(ConvBlock(3, in, out), ConvBlock(3, out, out))
    downsample = MaxPool((2, 2), pad=SamePad())
    EncoderBlock(conv_block, downsample)
end

function (l::EncoderBlock)(x)
    skip = l.conv_block(x)
    return l.downsample(skip), skip
end

struct DecoderBlock
    conv_block
    upsample
end

Flux.@functor DecoderBlock

function DecoderBlock(in::Int, out::Int)
    conv_block = Chain(ConvBlock(3, in, out), ConvBlock(3, out, out))
    upsample = Upsample(:bilinear, scale=(2, 2))
    DecoderBlock(conv_block, upsample)
end

function (l::DecoderBlock)(x, skip)
    return @pipe l.upsample(x) |> cat(_, skip, dims=3) |> l.conv_block
end

struct UNet
    encoder_blocks
    decoder_blocks
    activation
end

Flux.@functor UNet

function UNet(channels::Int, nclasses::Int, filters=[32, 64, 128, 256, 512])
    layers = length(filters)
    encoder_filters = vcat(channels, filters)
    encoder_blocks = [EncoderBlock(encoder_filters[i], encoder_filters[i+1]) for i in 1:layers]
    decoder_blocks = [DecoderBlock(filters[i]+filters[i+1], filters[i]) for i in 1:layers-1]
    activation = Chain(Conv((3, 3), filters[1]=>nclasses, pad=SamePad()), x -> softmax(x, dims=3))
    UNet(encoder_blocks, decoder_blocks, activation)
end

function (l::UNet)(x)
    # Forward Pass Through Encoder
    skips = nil()
    layers = length(l.encoder_blocks)
    for i in 1:layers-1
        x, skip = l.encoder_blocks[i](x)
        skips = cons(skip, skips)
    end
    _, x = l.encoder_blocks[end](x)

    # Forward Pass Through Decoder
    for (i, skip) in enumerate(skips)
        x = l.decoder_blocks[layers-i](x, skip)
    end

    # Activation Layer
    return l.activation(x)
end