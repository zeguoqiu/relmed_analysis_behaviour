# This file contains a Q learning model for single participant data, 
# and methods for sampling, fitting, and computing quanitities based on it.

# Turing model
@model function single_p_QL(;
	N::Int64, # Total number of trials
	n_blocks::Int64, # Number of blocks
	block::Vector{Int64}, # Block number
	valence::AbstractVector, # Valence of each block
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	outcomes::Matrix{Float64}, # Outcomes for options, second column optimal
	initV::Matrix{Float64}, # Initial Q values,
	σ_ρ::Float64 = 1.,
	σ_a::Float64 = 0.5
)

	# Priors on parameters
	ρ ~ truncated(Normal(0., σ_ρ), lower = 0.)
	a ~ Normal(0., σ_a)

	# Compute learning rate
	α = a2α(a) # hBayesDM uses Phi_approx from Stan. Here, logistic with the variance of the logistic multiplying a to equate the scales to that of a probit function.

	# Initialize Q values
	Qs = repeat(initV .* ρ, length(block)) .* valence[block]

	# Loop over trials, updating Q values and incrementing log-density
	for i in 1:N
		
		# Define choice distribution
		choice[i] ~ BernoulliLogit(Qs[i, 2] - Qs[i, 1])

		choice_idx::Int64 = choice[i] + 1

		# Prediction error
		PE = outcomes[i, choice_idx] * ρ - Qs[i, choice_idx]

		# Update Q value
		if (i != N) && (block[i] == block[i+1])
			Qs[i + 1, choice_idx] = Qs[i, choice_idx] + α * PE
			Qs[i + 1, 3 - choice_idx] = Qs[i, 3 - choice_idx]
		end
	end

	return Qs

end


# Simulate data from model prior
function simulate_single_p_QL(
	n::Int64; # How many datasets to simulate
	block::Vector{Int64}, # Block number
	valence::AbstractVector, # Valence of each block
	outcomes::Matrix{Float64}, # Outcomes for options, first column optimal
	initV::Matrix{Float64}, # Initial Q values
	random_seed::Union{Int64, Nothing} = nothing,
	σ_ρ::Float64 = 2.,
	σ_a::Float64 = 1.
)

	# Total trial number
	N = length(block)

	# Trials per block
    n_trials = div(length(block), maximum(block))

	# Prepare model for simulation
	prior_model = single_p_QL(
		N = N,
		n_blocks = maximum(block),
		block = block,
		valence = valence,
		choice = fill(missing, length(block)),
		outcomes = outcomes,
		initV = initV,
		σ_ρ = σ_ρ,
		σ_a = σ_a
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
		block = repeat(block, n),
		valence = repeat(valence, inner = n_trials, outer = n),
		trial = repeat(1:n_trials, n * maximum(block)),
		choice = prior_sample[:, [Symbol("choice[$i]") for i in 1:N], 1] |>
			Array |> transpose |> vec
	)

	# Compute Q values
	Qs = generated_quantities(prior_model, prior_sample) |> vec

	sim_data.Q_optimal = vcat([qs[:, 2] for qs in Qs]...) 
	sim_data.Q_suboptimal = vcat([qs[:, 1] for qs in Qs]...) 

	return sim_data
			
end

# Sample from posterior conditioned on DataFrame with data for single participant
function posterior_sample_single_p_QL(
	data::AbstractDataFrame;
	initV::Float64,
	random_seed::Union{Int64, Nothing} = nothing,
	iter_sampling = 1000,
	σ_ρ::Float64 = 2.,
	σ_a::Float64 = 1.
)
	model = single_p_QL(;
		N = nrow(data),
		n_blocks = maximum(data.block),
		block = data.block,
		valence = unique(data[!, [:block, :valence]]).valence,
		choice = data.choice,
		outcomes = hcat(
			data.feedback_suboptimal,
			data.feedback_optimal,
		),
		initV = fill(initV, 1, 2),
		σ_ρ = σ_ρ,
		σ_a = σ_a
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

# Find MLE / MAP for DataFrame with data for single participant
function optimize_single_p_QL(
	data::AbstractDataFrame;
	initV::Float64,
	estimate::String = "MAP",
	initial_params::Union{AbstractVector,Nothing}=nothing,
	σ_ρ::Float64 = 2.,
	σ_a::Float64 = 1.
)
	model = single_p_QL(;
		N = nrow(data),
		n_blocks = maximum(data.block),
		block = data.block,
		valence = unique(data[!, [:block, :valence]]).valence,
		choice = data.choice,
		outcomes = hcat(
			data.feedback_suboptimal,
			data.feedback_optimal,
		),
		initV = fill(initV, 1, 2),
		σ_ρ = σ_ρ,
		σ_a = σ_a
	)

	if estimate == "MLE"
		fit = maximum_likelihood(model; initial_params = initial_params)
	elseif estimate == "MAP"
		fit = maximum_a_posteriori(model; initial_params = initial_params)
	end

	return fit
end

# Find MLE / MAP multiple times
function optimize_multiple_single_p_QL(
	data::DataFrame;
	initV::Float64,
	estimate::String = "MAP",
	initial_params::Union{AbstractVector,Nothing}=[mean(truncated(Normal(0., 2.), lower = 0.)), 0.5],
	include_true::Bool = false, # Whether to return true value if this is simulation
	σ_ρ::Float64 = 2.,
	σ_a::Float64 = 1.
)
	ests = []
	lk = ReentrantLock()

	Threads.@threads for p in unique(data.PID)

		# Select data
		gdf = filter(x -> x.PID == p, data)

		# Optimize
		est = optimize_single_p_QL(
			gdf; 
			initV = initV,
			estimate = estimate,
			initial_params = initial_params,
			σ_ρ = σ_ρ,
			σ_a = σ_a
		)

		# Return
		if include_true
			est = (
				PID = gdf.PID[1],
				true_a = α2a(gdf.α[1]),
				true_ρ = gdf.ρ[1],
				MLE_a = est.values[:a],
				MLE_ρ = est.values[:ρ]
				)
		else
			est = (
				PID = gdf.PID[1],
				a = est.values[:a],
				ρ = est.values[:ρ]
			)
		end

		lock(lk) do
			push!(ests, est)
		end
	end

	return DataFrame(ests)
end

function bootstrap_optimize_single_p_QL(
	PLT_data::DataFrame;
	n_bootstrap::Int64 = 20)
	
		# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	prolific_pids = sort(unique(PLT_data.prolific_pid))

	bootstraps = []

	for i in 1:n_bootstrap
		
		# Resample the data with replacement
		idxs = sample(Xoshiro(i), prolific_pids, length(prolific_pids), replace=true)

		tdata = filter(x -> x.prolific_pid in prolific_pids, PLT_data)

		forfit, pids = prepare_for_fit(PLT_data)
		
		tfit = optimize_multiple_single_p_QL(
				forfit;
				initV = aao,
			)

		tfit = innerjoin(tfit, pids, on = :PID)

		tfit[!, :bootstrap_idx] .= i
		
		# Compute the correlation for the resampled data
		push!(bootstraps, tfit)
	end

	return vcat(bootstraps...)

end

# Sample from posterior for multiple datasets drawn for prior and summarise for simulation-based calibration
function SBC_single_p_QL(
	data::DataFrame;
	initV::Float64,
	random_seed::Union{Int64, Nothing} = nothing,
	iter_sampling = 500,
	σ_ρ::Float64 = 2.,
	σ_a::Float64 = 1.
)

	sums = []
	for p in unique(data.PID)
		gdf = filter(x -> x.PID == p, data)
		
		draws = posterior_sample_single_p_QL(
			gdf;
			initV = initV,
			random_seed = random_seed,
			iter_sampling = iter_sampling,
			σ_ρ = σ_ρ,
			σ_a = σ_a
		)

		push!(
			sums,
			sum_SBC_draws(
				draws,
				params = [:a, :ρ],
				true_values = [α2a(gdf.α[1]), gdf.ρ[1]],
				prior_var = [1., var(truncated(Normal(0., 2.), lower = 0.))]
			)
		)
	end
	
	return sums
end

# Simulate choice from posterior 
function simulate_from_posterior_single_p_QL(
	bootstrap::DataFrameRow, # DataFrameRow containinng parameters and condition
	task, # Task structure for condition,
	random_seed::Int64
) 

	# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	block = task.block
	valence = task.valence
	outcomes = task.outcomes

	post_model = single_p_QL(
		N = length(block),
		n_blocks = maximum(block),
		block = block,
		valence = valence,
		choice = fill(missing, length(block)),
		outcomes = outcomes,
		initV = fill(aao, 1, 2)
	)
	
	chn = Chains([bootstrap.a bootstrap.ρ], [:a, :ρ])

	choice = predict(Xoshiro(random_seed), post_model, chn)[1, :, 1] |> Array |> vec
	
	ppc = insertcols(task.task, 
		:isOptimal => choice,
		:bootstrap_idx => fill(bootstrap.bootstrap_idx, length(choice)),
		:prolific_pid => fill(bootstrap.prolific_pid, length(choice))
	)
	

end

# Prepare pilot data for fititng with model
function prepare_for_fit(data)

	forfit = select(data, [:prolific_pid, :condition, :session, :block, :valence, :trial, :optimalRight, :outcomeLeft, :outcomeRight, :chosenOutcome, :isOptimal])

	rename!(forfit, :isOptimal => :choice)

	# Make sure block is numbered correctly
	renumber_block(x) = indexin(x, sort(unique(x)))
	DataFrames.transform!(
		groupby(forfit, [:prolific_pid, :session]),
		:block => renumber_block => :block
	)

	# Arrange feedback by optimal / suboptimal
	forfit.feedback_optimal = 
		ifelse.(forfit.optimalRight .== 1, forfit.outcomeRight, forfit.outcomeLeft)

	forfit.feedback_suboptimal = 
		ifelse.(forfit.optimalRight .== 0, forfit.outcomeRight, forfit.outcomeLeft)

	# PID as number
	pids = unique(forfit[!, [:prolific_pid, :condition]])

	pids.PID = 1:nrow(pids)

	forfit = innerjoin(forfit, pids[!, [:prolific_pid, :PID]], on = :prolific_pid)

	# Block as Int64
	forfit.block = convert(Vector{Int64}, forfit.block)

	return forfit, pids
end
