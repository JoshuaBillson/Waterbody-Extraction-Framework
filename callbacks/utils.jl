using Plots, Images
using Pipe: @pipe

"""
### Description
Plot a single image to the given size.

# Arguments
`img`: The image we want to plot.  
`sz`: The size (width, height) of the image in the plot.  
"""
function showimg(img::Any, sz::Tuple{Int,Int})
	plot(img, size=sz, axis=nothing, showaxis=false, margin=0Plots.mm)
end

"""
# Description
Plot a collection of images, all of which are of the same size.

# Arguments
`imgs`: The vector of images which we want to plot.  
`sz`: The size (width, height) of the images. All images will be plotted to the same size.  
`layout`: The layout of the images as (rows, columns).  
"""
function showimg(imgs::AbstractVector, sz::Tuple{Int,Int}, layout::Tuple{Int,Int})
	plot(imgs..., layout=layout, size=sz, axis=nothing, showaxis=false, margin=0Plots.mm)
end

"""
Plot an RGB tile stored as an Array{Float32,3}.

`img`: The img as an tensor with the shape (HxWx3).  
`gamma`: Used to apply gamma correction to the image before plotting.
"""
function plot_color(img::Array{Float32,3}, gamma=1.0)
	scale = @pipe findmax(img)[1] |> Float32 |> max(_, 1.0f0)
	@pipe img .|> 
	/(_, scale) |>
	permutedims(_, (3, 2, 1)) |> 
	colorview(RGB, _) |> 
	adjust_histogram(_, GammaCorrection(gamma=gamma)) |>
	showimg(_, (1000, 1000))
end;

"""
Plot a grayscale tile stored as an Array{Float32,3}.

`img`: The img as an tensor with the shape (HxWx3).  
"""
function plot_gray(img::Array{Float32,3})
	scale = @pipe findmax(img)[1] |> Float32 |> max(_, 1.0f0)
	@pipe img[:,:,1] |> 
	/(_, scale) |>
	permutedims(_, (2, 1)) |> 
	colorview(Gray, _) |>
	showimg(_, (1000, 1000))
end;