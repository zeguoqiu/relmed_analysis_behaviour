### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ ed4ee070-530e-11ef-143e-09c737b06d9b
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase,
		ForwardDiff, LinearAlgebra, Memoization, LRUCache, GLM, JLD2, FileIO, JuMP, CSV, Dates, JSON, RCall, HTTP
	using IterTools: product
	using LogExpFunctions: logsumexp
	using Combinatorics: combinations

	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/stan_functions.jl")
	include("$(pwd())/PLT_task_functions.jl")
	include("$(pwd())/fisher_information_functions.jl")
	include("$(pwd())/plotting_functions.jl")

end

# ╔═╡ 0136c52b-019f-490a-87b5-487d2a91414f
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

# ╔═╡ 1fdd69b2-8b02-428a-a7a5-3e7c76656cf5
# Load and clean data
begin
	PLT_data = load_PLT_data()

	PLT_data = exclude_PLT_sessions(PLT_data)

	nothing
end

# ╔═╡ b64cabeb-f82b-4ce7-af56-63c8f02c44de
# Functions needed for data fit
begin
	# Prepare data for q learning model
	function prepare_data_for_fit(data::DataFrame)

		# Sort
		forfit = sort(data, [:condition, :prolific_pid, :block, :trial])
	
		# Make numeric pid variable
		pids = DataFrame(prolific_pid = unique(forfit.prolific_pid))
		pids.pp = 1:nrow(pids)
	
		forfit = innerjoin(forfit, pids, on = :prolific_pid)
	
		# Sort. If more than one session - renumber blocks
		if length(unique(data.session)) > 1
			forfit.cblock = (parse.(Int64, forfit.session) .- 1) .* 
				maximum(forfit.block) .+ forfit.block
			sort!(forfit, [:pp, :cblock, :trial])	
		else
			sort!(forfit, [:pp, :block, :trial])	
		end
		
		
		@assert all(combine(groupby(forfit, [:prolific_pid, :session, :block]),
			:trial => issorted => :trial_sorted
			).trial_sorted) "Trials not sorted"

		@assert all(combine(groupby(forfit, [:prolific_pid, :session]),
			:block => issorted => :block_sorted
		).block_sorted)  "Blocks not sorted"

		return forfit, pids

	end

	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
end

# ╔═╡ 91759c41-a311-4861-9868-6665b31b88da
md"""
# Full dataset test-retest
"""

# ╔═╡ d05df98e-2717-4f24-bfc6-4af22885e276
# Fit session 1
begin
	# Filter by session
	sess1_data = filter(x -> x.session == "1", PLT_data)

	# Prepare
	sess1_forfit, sess1_pids = prepare_data_for_fit(sess1_data)

	m1s1_sum, m1s1_draws, m1s1_time = load_run_cmdstanr(
		"m1s1",
		"group_QLRs02.stan",
		to_standata(sess1_forfit,
			aao;
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3
	)
	m1s1_sum, m1s1_time
end

# ╔═╡ dbb31f0d-b0ce-4c55-b932-d51a249e248e
# Fit session 2
begin
	# Filter by session
	sess2_data = filter(x -> x.session == "2", PLT_data)

	# Prepare
	sess2_forfit, sess2_pids = prepare_data_for_fit(sess2_data)

	m1s2_sum, m1s2_draws, m1s2_time = load_run_cmdstanr(
		"m1s2",
		"group_QLRs02.stan",
		to_standata(sess2_forfit,
			aao;
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3
	)
	m1s2_sum, m1s2_time
end

# ╔═╡ Cell order:
# ╠═ed4ee070-530e-11ef-143e-09c737b06d9b
# ╠═0136c52b-019f-490a-87b5-487d2a91414f
# ╠═1fdd69b2-8b02-428a-a7a5-3e7c76656cf5
# ╠═b64cabeb-f82b-4ce7-af56-63c8f02c44de
# ╟─91759c41-a311-4861-9868-6665b31b88da
# ╠═d05df98e-2717-4f24-bfc6-4af22885e276
# ╠═dbb31f0d-b0ce-4c55-b932-d51a249e248e
