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
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase, JSON, CSV, HTTP

	include("fetch_preprocess_data.jl")
end

# ╔═╡ Cell order:
# ╠═e01188c3-ca30-4a7c-9101-987752139a71
