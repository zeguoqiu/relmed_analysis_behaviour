# This file contains a Q learning model for data for group data, with no hierarchy - 
# each participant treated as independent of the rest. This is useful for ML/PML estimation.
# The file also contains methods for sampling, fitting, and computing quanitities based on it.

# Turing model
@model function independent_group_QL(;
	# All data input is vectorized with an element per participant
	block::Vector{Vector{Int64}}, # Block number per trial
	valence::Vector{Vector{Float64}}, # Valence of each block. Vector of lenth maximum(block) per participant
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	outcomes::Vector{Matrix{Float64}}, # Outcomes for options, second column optimal
	initV::Matrix{Float64} # Initial Q values
)   

    # Auxillary variables
    n_p = length(block) # Number of participants

	# Priors on parameters
	ρ ~ filldist( truncated(Normal(0., 2.), lower = 0.), n_p)
	a ~ MvNormal(fill(0., n_p), I)

	# Compute learning rate
	α = a2α.(a) # hBayesDM uses Phi_approx from Stan. Here, logistic with the variance of the logistic multiplying a to equate the scales to that of a probit function.

	# Initialize Q values per participant
	Qs = [initV .* (ρ[s] .* valence[s][block[s]]) for s in eachindex(valence)]

    # Loop over participants
    for s in eachindex(block)

        # Loop over trials, updating Q values and incrementing log-density
        for i in eachindex(block[s])
            
            # Define choice distribution
            choice[s][i] ~ BernoulliLogit(Qs[s][i, 2] - Qs[s][i, 1])

            choice_idx::Int64 = choice[s][i] + 1

            # Prediction error
            PE = outcomes[s][i, choice_idx] * ρ[s] - Qs[s][i, choice_idx]

            # Update Q value
            if (i != length(block[s])) && (block[s][i] == block[s][i+1])
                Qs[s][i + 1, choice_idx] = Qs[s][i, choice_idx] + α[s] * PE # Chosen updated
                Qs[s][i + 1, 3 - choice_idx] = Qs[s][i, 3 - choice_idx] # Unchosen carried over as is
            end
        end
    end

	return (choice = choice, Qs = Qs)

end

# Simulate data from model prior
function simulate_independent_group_QL(
	n::Int64; # How many datasets to simulate
	block::Vector{Vector{Int64}}, # Block number
	valence::Vector{Vector{Float64}}, # Valence of each block
	outcomes::Vector{Matrix{Float64}}, # Outcomes for options, first column optimal
	initV::Matrix{Float64}, # Initial Q values
	random_seed::Union{Int64, Nothing} = nothing
)

    # Check lengths
    @assert length(block) == lenght(valence) "Number of participants not consistent"
    @assert length(block) == lenght(outcomes) "Number of participants not consistent"
    @assert all([length(b) for b in block] .== [size(o, 1) for o in outcomes]) "Number of trials not consistent"
    @assert all[maximum(block[s]) == length(valence[s] for s in eachindex(block))] "Number of blocks not consistent"

	# Prepare model for simulation
	prior_model = independent_group_QL(
		block = block,
		valence = valence,
		choice = [fill(missing, length(block[s])) for s in eachindex(block)],
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

	# Extract parameters and choices
    ρ = single_layer_chain_to_vector(prior_sample, "ρ")
    a = single_layer_chain_to_vector(prior_sample, "a")

	# Compute Q values and choice
	gq = generated_quantities(prior_model, prior_sample) |> vec

    # Exctract choice
    choice = [d[:choice] for d in gq]

    # Exctract Qs
    Qs = [d[:Qs] for d in gq]

	return (
        ρ = ρ,
        a = a,
        choice = choices,
        Qs = Qs
    )
			
end

# Exctract vector from chains
function single_layer_chain_to_vector(
    chain,
    param::String
)   
    # Find columns
    matching_columns = filter(x -> occursin(Regex("$(param)\\[\\d+\\]"), x), string.(names(chain, :parameters)))

    # Extract values
    outcome =  [vec(Array(chain[i, matching_columns, :])) for i in 1:size(chain, 1)]

    # Check lengths match
    @assert allequal(length.(outcome))

    # Reutrn flat if 1 iteration only
    if length(outcome) == 1
        return outcome[1]
    else
        return outcome
    end
end

