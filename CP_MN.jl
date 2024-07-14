### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ e01188c3-ca30-4a7c-9101-987752139a71
begin
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase, JSON, CSV, HTTP, RCall, JLD2

	include("fetch_preprocess_data.jl")
	include("stan_functions.jl")
end

# ╔═╡ bdeadcc4-1a5f-4c39-a055-e61b3db3f3b1
begin
	# Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")
	
	th = Theme(
		font = "Helvetica",
		fontsize = 16,
		Axis = (
			xgridvisible = false,
			ygridvisible = false,
			rightspinevisible = false,
			topspinevisible = false,
			xticklabelsize = 14,
			yticklabelsize = 14,
			spinewidth = 1.5,
			xtickwidth = 1.5,
			ytickwidth = 1.5
		)
	)
	set_theme!(th)
end

# ╔═╡ 963e5f75-00f9-4fcc-90b9-7ecfb7e278f2
# Load data
PLT_data = load_PLT_data()

# ╔═╡ Cell order:
# ╠═e01188c3-ca30-4a7c-9101-987752139a71
# ╠═bdeadcc4-1a5f-4c39-a055-e61b3db3f3b1
# ╠═963e5f75-00f9-4fcc-90b9-7ecfb7e278f2
