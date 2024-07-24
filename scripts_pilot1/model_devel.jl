### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 8c7452ce-49c8-11ef-2441-d5bcc4726e41
begin
	cd("/home/jovyan/")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase,
		ForwardDiff, LinearAlgebra, Memoization, LRUCache, GLM, JLD2, FileIO, JuMP, CSV, Dates, JSON, RCall
	using IterTools: product
	using LogExpFunctions: logsumexp
	using Combinatorics: combinations

	include("fetch_preprocess_data.jl")
	include("stan_functions.jl")
	include("PLT_task_functions.jl")
	include("fisher_information_functions.jl")
	include("plotting_functions.jl")

end

# ╔═╡ 1645ffed-f945-4cb4-9e26-fa7ec40117aa
# Load and clean data
begin
	PLT_data = load_PLT_data()

	PLT_data = exclude_PLT_sessions(PLT_data)

	nothing
end

# ╔═╡ c3a58ad1-3a58-4689-b308-3d17b5325e22
# Functions needed for data fit
begin
	# Function to get initial values
	function initV(data::DataFrame)

		# List of blocks
		blocks = unique(data[!, [:pp, :session, :block, :valence, :valence_grouped]])

		# Absolute mean reward for grouped
		amrg = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
		
		# Valence times amrg
		initVs = blocks.valence .* amrg

		return [fill(i, 2) for i in initVs]
	end

	# Prepare data for q learning model
	function prepare_data_for_fit(data::DataFrame)

		# Sort
		forfit = sort(data, [:condition, :prolific_pid, :block, :trial])
	
		# Make numeric pid variable
		pids = DataFrame(prolific_pid = unique(forfit.prolific_pid))
		pids.pp = 1:nrow(pids)
	
		forfit = innerjoin(forfit, pids, on = :prolific_pid)
	
		# Sort
		if length(unique(data.session)) > 1
			forfit.cblock = (parse.(Int64, forfit.session) .- 1) .* 
				maximum(forfit.block) .+ forfit.block
			sort!(forfit, [:pp, :cblock, :trial])	
		else
			sort!(forfit, [:pp, :block, :trial])	
		end
		
		
		@assert length(initV(forfit)) == 
			nrow(unique(forfit[!, [:prolific_pid, :session, :block]])) "initV does not return a vector with length n_total_blocks"

		@assert all(combine(groupby(forfit, [:prolific_pid, :session, :block]),
			:trial => issorted => :trial_sorted
			).trial_sorted) "Trials not sorted"

		@assert all(combine(groupby(forfit, [:prolific_pid, :session]),
			:block => issorted => :block_sorted
		).block_sorted)  "Blocks not sorted"

		return forfit, pids

	end
end

# ╔═╡ 821fe42b-c46d-4cc4-89b4-af23637c01e4
# Fit to all data
begin
	# Prepare
	forfit, pids = prepare_data_for_fit(PLT_data)

	m1_sum, m1_draws, m1_time = load_run_cmdstanr(
		"m1",
		"group_QLrs02.stan",
		to_standata(forfit,
			initV;
			block_col = :cblock,
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3
	)
	m1_sum, m1_time
end

# ╔═╡ 140f37e7-b345-4358-8ab2-62ff318f8758
forfit[!, [:prolific_pid, :session, :block, :cblock, :trial]]

# ╔═╡ 08ba9b22-bb8d-4a0f-bbb9-277f089cc119
begin
	# Filter by session
	sess1_data = filter(x -> x.session == "1", PLT_data)

	# Prepare
	sess1_forfit, sess1_pids = prepare_data_for_fit(sess1_data)

	m1s1_sum, m1s1_draws, m1s1_time = load_run_cmdstanr(
		"m1s1",
		"group_QLrs.stan",
		to_standata(sess1_forfit,
			initV;
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3,
		load_model = true
	)
	m1s1_sum, m1s1_time
end

# ╔═╡ d2b13736-503e-421f-80d7-9f6d8156b0e9
begin
	# Filter by session
	sess2_data = filter(x -> x.session == "2", PLT_data)

	# Prepare
	sess2_forfit, sess2_pids = prepare_data_for_fit(sess2_data)

	m1s2_sum, m1s2_draws, m1s2_time = load_run_cmdstanr(
		"m1s2",
		"group_QLrs.stan",
		to_standata(sess2_forfit,
			initV;
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3,
		load_model = true
	)
	m1s2_sum, m1s2_time
end

# ╔═╡ cc321700-89fe-4867-ad0a-3d6af980219c
begin
	m1s2_test_sum, m1s2_test_draws, m1s2_test_time = load_run_cmdstanr(
		"m1s2_test",
		"group_QLrs02.stan",
		to_standata(sess2_forfit,
			initV;
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3,
		load_model = true
	)
	m1s2_test_sum, m1s2_test_time
end

# ╔═╡ Cell order:
# ╠═8c7452ce-49c8-11ef-2441-d5bcc4726e41
# ╠═1645ffed-f945-4cb4-9e26-fa7ec40117aa
# ╠═c3a58ad1-3a58-4689-b308-3d17b5325e22
# ╠═821fe42b-c46d-4cc4-89b4-af23637c01e4
# ╠═140f37e7-b345-4358-8ab2-62ff318f8758
# ╠═08ba9b22-bb8d-4a0f-bbb9-277f089cc119
# ╠═d2b13736-503e-421f-80d7-9f6d8156b0e9
# ╠═cc321700-89fe-4867-ad0a-3d6af980219c
