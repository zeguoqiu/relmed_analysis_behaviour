### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ ac7fd352-555d-11ef-0f98-07f8c7c23d25
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase,
		ForwardDiff, LinearAlgebra, Memoization, LRUCache, GLM, JLD2, FileIO, JuMP, CSV, Dates, JSON, RCall, Turing, ParetoSmooth, MCMCDiagnosticTools
	using IterTools: product
	using LogExpFunctions: logsumexp, logistic
	using Combinatorics: combinations

	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/stan_functions.jl")
	include("$(pwd())/PLT_task_functions.jl")
	include("$(pwd())/fisher_information_functions.jl")
	include("$(pwd())/plotting_functions.jl")

end

# ╔═╡ ae31ab2d-3d59-4c33-b74a-f21dd43ca95e
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

# ╔═╡ 76412db2-9b4c-4aea-a049-3831321347ab
# Load and clean data
begin
	PLT_data = load_PLT_data()

	PLT_data = exclude_PLT_sessions(PLT_data)

	nothing
end

# ╔═╡ 43d7b28a-97a3-4db7-9e41-a7e73aa18b81
md"""
## Single Q learner
"""

# ╔═╡ 07492d7a-a15a-4e12-97b6-1e85aac23e4f
@model function single_p_QL(;
	N::Int64, # Total number of trials
	n_blocks::Int64, # Number of blocks
	n_trials::Int64, # Number of trials in block
	bl::Vector{Int64}, # Block number
	valence::AbstractVector, # Valence of each block
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	outcomes::Matrix{Float64}, # Outcomes for options, second column optimal
	initV::Matrix{Float64} # Initial Q values
)

	# Priors on parameters
	ρ ~ truncated(Normal(0., 2.), lower = 0.)
	a ~ Normal(0., 1.)

	# Compute learning rate
	α = logistic(π/sqrt(3) * a) # hBayesDM uses Phi_approx from Stan. Here, logistic with the variance of the logistic multiplying a to equate the scales to that of a probit function.

	# Initialize Q values
	Qs = vcat([repeat(initV .* (ρ * valence[i]), n_trials) for i in 1:n_blocks]...)

	# Loop over trials, updating Q values and incrementing log-density
	for i in 1:N
		
		# Define choice distribution
		choice[i] ~ BernoulliLogit(Qs[i, 1] - Qs[i, 2])

		choice_idx::Int64 = choice[i] + 1

		# Prediction error
		PE = outcomes[i, choice_idx] * ρ - Qs[i, choice_idx]

		# Update Q value
		if (i != N) && (bl[i] == bl[i+1])
			Qs[i + 1, choice_idx] = Qs[i, choice_idx] + α * PE
			Qs[i + 1, 3 - choice_idx] = Qs[i, 3 - choice_idx]
		end
	end

	return Qs

end

# ╔═╡ 842bda72-9c09-4170-a40c-04d2510b7673
function fit_to_df_single_p_QL(
	data::AbstractDataFrame;
	initV::Float64,
	random_seed::Union{Int64, Nothing} = nothing,
	iter_sampling = 1000
)
	model = single_p_QL(;
		N = nrow(data),
		n_blocks = maximum(data.block),
		n_trials = maximum(data.trial),
		bl = data.block,
		valence = unique(data[!, [:block, :valence]]).valence,
		choice = data.choice,
		outcomes = hcat(
			data.feedback_suboptimal,
			data.feedback_optimal,
		),
		initV = fill(initV, 1, 2)
	)

	fit = sample(
		isnothing(random_seed) ? Random.default_rng() : Xoshiro(random_seed),
		model, 
		NUTS(), 
		MCMCThreads(), 
		iter_sampling, 
		4)

	return fit

end

# ╔═╡ ef1246c9-6a0c-49e7-aaa4-f5dbac769cc8
function fit_sum_multiple_single_p_QL(
	data::DataFrame;
	initV::Float64,
	random_seed::Union{Int64, Nothing} = nothing,
	iter_sampling = 500
)

	sums = []
	for p in unique(data.PID)
		gdf = filter(x -> x.PID == p, data)
		
		draws = fit_to_df_single_p_QL(
			gdf;
			initV = initV,
			random_seed = random_seed,
			iter_sampling = iter_sampling
		)

		push!(
			sums,
			sum_prior_predictive_draws(
				draws,
				params = [:a, :ρ],
				true_values = [α2a(gdf.α[1]), gdf.ρ[1]],
				prior_var = [1., var(truncated(Normal(0., 2.), lower = 0.))]
			)
		)
	end
	
	return sums
end

# ╔═╡ 69f78ddd-2310-4534-997c-6888e2808ea5
function simulate_single_p_QL(
	n::Int64; # How many datasets to simulate
	bl::Vector{Int64}, # Block number
	valence::AbstractVector, # Valence of each block
	outcomes::Matrix{Float64}, # Outcomes for options, first column optimal
	initV::Matrix{Float64}, # Initial Q values
	random_seed::Union{Int64, Nothing} = nothing
)

	# Total trial number
	N = length(bl)

	# Trials per block
	n_trials = div(length(bl), maximum(bl))

	# Prepare model for simulation
	prior_model = single_p_QL(
		N = N,
		n_trials = n_trials,
		n_blocks = maximum(bl),
		bl = bl,
		valence = valence,
		choice = fill(missing, length(bl)),
		outcomes = outcomes,
		initV = initV
	)


	# Draw parameters and simulate choice
	prior_sample = sample(
		isnothing(random_seed) ? Random.default_rng() : Xoshiro(random_seed),
		prior_model,
		Prior(),
		n
	)

	# Arrange choice for return
	sim_data = DataFrame(
		PID = repeat(1:n, inner = N),
		ρ = repeat(prior_sample[:, :ρ, 1], inner = N),
		α = repeat(prior_sample[:, :a, 1], inner = N) .|> a2α,
		block = repeat(bl, n),
		valence = repeat(valence, inner = n_trials, outer = n),
		trial = repeat(1:n_trials, n * maximum(bl)),
		choice = prior_sample[:, [Symbol("choice[$i]") for i in 1:N], 1] |>
			Array |> transpose |> vec
	)

	# Compute Q values
	Qs = generated_quantities(prior_model, prior_sample) |> vec

	sim_data.Q_optimal = vcat([qs[:, 2] for qs in Qs]...) 
	sim_data.Q_suboptimal = vcat([qs[:, 1] for qs in Qs]...) 

	return sim_data
			
end

# ╔═╡ 14b82fda-229b-4a52-bc49-51201d4706be
prior_sample = let
	# Load sequence from file
	task = DataFrame(CSV.File("data/PLT_task_structure_00.csv"))

	# Renumber block
	task.block = task.block .+ (task.session .- 1) * maximum(task.block)

	# Arrange feedback by optimal / suboptimal
	task.feedback_optimal = 
		ifelse.(task.optimal_A .== 1, task.feedback_A, task.feedback_B)

	task.feedback_suboptimal = 
		ifelse.(task.optimal_A .== 0, task.feedback_A, task.feedback_B)


	# Arrange outcomes such as second column is optimal
	outcomes = hcat(
		task.feedback_suboptimal,
		task.feedback_optimal,
	)

	# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	prior_sample = simulate_single_p_QL(
		100;
		bl = task.block,
		valence = unique(task[!, [:block, :valence]]).valence,
		outcomes = outcomes,
		initV = fill(aao, 1, 2),
		random_seed = 0
	)


	leftjoin(prior_sample, 
		task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
		on = [:block, :trial]
	)

end

# ╔═╡ b0e6dde1-c740-4426-a4a1-b609aa6af536
sbc = let
	draws_sum_file = "saved_models/sbc_single_p_QL.jld2"
	if isfile(draws_sum_file)
		JLD2.@load draws_sum_file sbc
	else
		sbc = fit_sum_multiple_single_p_QL(
			prior_sample;
			initV = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])]),
			random_seed = 0
		) |> DataFrame

		JLD2.@save draws_sum_file sbc
	end

	sbc
end

# ╔═╡ 90e25dd4-12be-410b-9241-83f8a4d417d5
plot_prior_predictive(sbc, show_n = [1], params = ["a", "ρ"])

# ╔═╡ 24f05d41-c1a8-4251-a4f5-bab478b7f1f0
begin
	fit = let
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
		
		fit = fit_to_df_single_p_QL(
			filter(x -> x.PID == 1, prior_sample); 
			initV = aao,
			random_seed = 0
		)
	
		fit
	
	end

	describe(fit)

end

# ╔═╡ 9e03d03e-f58f-4d8a-8a25-b0f1d9d1da0c
plot_posteriors([fit],
	["a", "ρ"];
	true_values = [α2a(prior_sample[1, :α]), prior_sample[1, :ρ]]
)	

# ╔═╡ 78b422d6-c70f-4a29-a433-7173e1b108a0
let
	df = rename(prior_sample, 
		:Q_optimal => :EV_A,
		:Q_suboptimal => :EV_B
	)

	df[!, :group] .= 1
	
	f = Figure(size = (700, 1000))

	g_all = f[1,1] = GridLayout()
	
	plot_sim_q_value_acc!(
		g_all,
		df;
		plw = 1,
		legend = false,
		acc_error_band = "PI"
	)

	g_reward = f[2,1] = GridLayout()
	
	plot_sim_q_value_acc!(
		g_reward,
		filter(x -> x.valence > 0, df);
		plw = 1,
		legend = false,
		acc_error_band = "PI"
	)

	g_reward = f[3,1] = GridLayout()
	
	plot_sim_q_value_acc!(
		g_reward,
		filter(x -> x.valence < 0, df);
		plw = 1,
		legend = false,
		acc_error_band = "PI"
	)

	f
end

# ╔═╡ 8e4f80e1-bdd8-4080-a64c-e4d703ef5544
typeof("a") == String

# ╔═╡ Cell order:
# ╠═ac7fd352-555d-11ef-0f98-07f8c7c23d25
# ╠═ae31ab2d-3d59-4c33-b74a-f21dd43ca95e
# ╠═76412db2-9b4c-4aea-a049-3831321347ab
# ╟─43d7b28a-97a3-4db7-9e41-a7e73aa18b81
# ╠═07492d7a-a15a-4e12-97b6-1e85aac23e4f
# ╠═ef1246c9-6a0c-49e7-aaa4-f5dbac769cc8
# ╠═90e25dd4-12be-410b-9241-83f8a4d417d5
# ╠═b0e6dde1-c740-4426-a4a1-b609aa6af536
# ╠═842bda72-9c09-4170-a40c-04d2510b7673
# ╠═24f05d41-c1a8-4251-a4f5-bab478b7f1f0
# ╠═9e03d03e-f58f-4d8a-8a25-b0f1d9d1da0c
# ╠═14b82fda-229b-4a52-bc49-51201d4706be
# ╠═78b422d6-c70f-4a29-a433-7173e1b108a0
# ╠═69f78ddd-2310-4534-997c-6888e2808ea5
# ╠═8e4f80e1-bdd8-4080-a64c-e4d703ef5544
