# This file contains a Q learning model for single participant data, 
# and methods for sampling, fitting, and computing quanitities based on it.

# Turing model
@model function single_p_QL(;
	N::Int64, # Total number of trials
	n_blocks::Int64, # Number of blocks
	n_trials::Int64, # Number of trials in block
	block::Vector{Int64}, # Block number
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
	random_seed::Union{Int64, Nothing} = nothing
)

	# Total trial number
	N = length(block)

	# Trials per block
	n_trials = div(length(block), maximum(block))

	# Prepare model for simulation
	prior_model = single_p_QL(
		N = N,
		n_trials = n_trials,
		n_blocks = maximum(block),
		block = block,
		valence = valence,
		choice = fill(missing, length(block)),
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
	iter_sampling = 1000
)
	model = single_p_QL(;
		N = nrow(data),
		n_blocks = maximum(data.block),
		n_trials = maximum(data.trial),
		block = data.block,
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

# Sample from posterior for multiple datasets drawn for prior and summarise for simulation-based calibration
function SBC_single_p_QL(
	data::DataFrame;
	initV::Float64,
	random_seed::Union{Int64, Nothing} = nothing,
	iter_sampling = 500
)

	sums = []
	for p in unique(data.PID)
		gdf = filter(x -> x.PID == p, data)
		
		draws = posterior_sample_single_p_QL(
			gdf;
			initV = initV,
			random_seed = random_seed,
			iter_sampling = iter_sampling
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