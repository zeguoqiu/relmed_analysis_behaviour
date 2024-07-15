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
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase, JSON, CSV, HTTP, RCall, JLD2, Dates

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
# Load and clean data
begin
	PLT_data = load_PLT_data()

	PLT_data = exclude_PLT_sessions(PLT_data)

end

# ╔═╡ db04f1f3-5c9f-4418-a9b9-bd3200f0d4c4
names(PLT_data)

# ╔═╡ fe070ddf-82cd-4c5f-8bb1-8adab53f654f
# Session 1 model
begin
	# Filter by session
	sess1_data = filter(x -> x.session == "1", PLT_data)

	# Sort
	sort!(sess1_data, [:condition, :prolific_pid, :block, :trial])

	@assert maximum(sess1_data.block) == 24 "Block numbers are not what you expect"

	# Function to get initial values
	function initV(data::DataFrame)

		# List of blocks
		blocks = unique(data[!, [:prolific_pid, :block, :valence, :valence_grouped]])

		# Absolute mean reward for grouped
		amrg = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
		
		# Valence times amrg
		initVs = blocks.valence .* amrg

		return initVs
	end

	@assert length(initV(sess1_data)) == nrow(unique(sess1_data[!, [:prolific_pid, :block]])) "initV does not return a vector with length n_total_blocks"

	

end

# ╔═╡ 78549c5d-48b4-4634-b380-b2b8d883d430
begin
	m1_sum, m1_draws, m1_time = load_run_cmdstanr(
		"m1",
		"group_QLrs.stan",
		to_standata(PLT_data,
			feedback_magnitudes,
			feedback_ns;
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3
	)
	group_QLrs_sum, group_QLrs_time
end

# ╔═╡ Cell order:
# ╠═e01188c3-ca30-4a7c-9101-987752139a71
# ╠═bdeadcc4-1a5f-4c39-a055-e61b3db3f3b1
# ╠═963e5f75-00f9-4fcc-90b9-7ecfb7e278f2
# ╠═db04f1f3-5c9f-4418-a9b9-bd3200f0d4c4
# ╠═fe070ddf-82cd-4c5f-8bb1-8adab53f654f
# ╠═78549c5d-48b4-4634-b380-b2b8d883d430
