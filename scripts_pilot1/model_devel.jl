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

# ╔═╡ eeaaea99-673d-4f34-a633-64955caa971e
function extract_participant_params(
	draws::DataFrame;
	params::Vector{String} = ["a", "rho"],
	rescale::Bool = true
) 

	res = Dict()

	for param in params

		# Select columns in draws DataFrame
		tdraws = select(draws, Regex("$(param)\\[\\d+\\]"))
	
		if rescale
			# Add mean and multiply by SD
			tdraws .*= draws[!, Symbol("sigma_$param")]
			tdraws .+= draws[!, Symbol("mu_$param")]	
		end
	
		rename!(s -> replace(s, Regex("$param\\[(\\d+)\\]") => s"\1"),
			tdraws
			)

		res[param] = tdraws
	end

	return res
end

# ╔═╡ cee9b208-46ad-4e31-92c4-d2bfdbad7ad3
function q_learning_posterior_predictive_draw(i;
	data::DataFrame = copy(forfit),
	draw::DataFrame,
	amr::Float64 = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])]) # Average mean reward per block
)
		
	# Keep only needed variables
	task = select(
		data,
		[:pp, :session, :block, :trial, :optimalRight, 
			:outcomeRight, :outcomeLeft, :valence, :early_stop]
	)
			
	# Join data with parameters
	task = leftjoin(task, draw, on = :pp)

	# Rearrange data to conform to option A is optimal
	task.feedback_A = ifelse.(
		task.optimalRight .== 1,
		task.outcomeRight,
		task.outcomeLeft
	)

	task.feedback_B = ifelse.(
		task.optimalRight .== 0,
		task.outcomeRight,
		task.outcomeLeft
	)

	# Compute initial values
	task.initV = task.valence .* amr

	# Stop after variable
	task.stop_after = ifelse.(task.early_stop, 5, missing)

	# Convenience function for simulation
	function simulate_grouped_block(grouped_df)
		simulate_block(grouped_df,
			2,
			grouped_df.initV[1] .* grouped_df.ρ[1], # Use average reward modulated by rho as initial Q value
			q_learning_w_rho_update,
			[:α, :ρ],
			softmax_choice_direct,
			Vector{Symbol}();
			stop_after = grouped_df.stop_after[1]
		)
	end

	# Simulate data per participants, block
	grouped_task = groupby(task, [:pp, :session, :block])
	sims = transform(grouped_task, simulate_grouped_block)

	sims.draw .= i

	return sims
end


# ╔═╡ 7d836234-8703-48d0-9c66-f39761a57d65
begin
	participant_params = extract_participant_params(m1_draws)

	Random.seed!(0)

	m1_ppc = []
	for i in sample(1:nrow(participant_params["a"]), 100)

		# Extract draw
		draw = DataFrame(
			α = a2α.(stack(participant_params["a"][1, :])),
			ρ = stack(participant_params["rho"][1, :])
		)
		draw.pp = 1:nrow(draw)

		# Simulate task
		ppd = q_learning_posterior_predictive_draw(i;
			data = forfit,
			draw = draw
		)

		push!(m1_ppc, ppd)
	end

	m1_ppc = vcat(m1_ppc...)
end

# ╔═╡ 1fc6a457-cd86-4835-a18a-75db1fe4a469
begin
	m1_ppc_sum = combine(
		groupby(m1_ppc, 
			[:pp, :draw, :trial]),
		:choice => (x -> mean(x .== 1)) => :isOptimal
	)

	m1_ppc_sum = combine(
		groupby(m1_ppc_sum, [:draw, :trial]),
		:isOptimal => mean => :isOptimal
	)

	m1_ppc_sum = combine(
		groupby(m1_ppc_sum, :trial),
		:isOptimal => median => :m,
		:isOptimal => lb => :lb,
		:isOptimal => llb => :llb,
		:isOptimal => ub => :ub,
		:isOptimal => uub => :uub
	)

	f_acc = Figure()

	# Plot data
	ax_acc = plot_group_accuracy!(f_acc[1,1], forfit;
		error_band = false
	)

	# Plot 

	f_acc
end

# ╔═╡ 0bbcf2ce-4297-4543-8659-94ea07b2e0f7
typeof(true)

# ╔═╡ Cell order:
# ╠═8c7452ce-49c8-11ef-2441-d5bcc4726e41
# ╠═1645ffed-f945-4cb4-9e26-fa7ec40117aa
# ╠═c3a58ad1-3a58-4689-b308-3d17b5325e22
# ╠═821fe42b-c46d-4cc4-89b4-af23637c01e4
# ╠═eeaaea99-673d-4f34-a633-64955caa971e
# ╠═cee9b208-46ad-4e31-92c4-d2bfdbad7ad3
# ╠═7d836234-8703-48d0-9c66-f39761a57d65
# ╠═1fc6a457-cd86-4835-a18a-75db1fe4a469
# ╠═0bbcf2ce-4297-4543-8659-94ea07b2e0f7
