### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ da2aa306-75f9-11ef-2592-2be549c73d82
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP
	include("fetch_preprocess_data.jl")
end

# ╔═╡ 6eba46dc-855c-47ca-8fa9-8405b9566809
jspsych_data, records = get_REDCap_data("pilot2"; file_field = "file_data")

# ╔═╡ Cell order:
# ╠═da2aa306-75f9-11ef-2592-2be549c73d82
# ╠═6eba46dc-855c-47ca-8fa9-8405b9566809
