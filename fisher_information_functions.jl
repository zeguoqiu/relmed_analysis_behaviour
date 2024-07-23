function Q_learning_log_likelihood(
	α, # Learning rate
	ρ, # Reward sensitivity
	R, # Reward on previous trial
	prev_choice, # Choice on previous trial,
	Q, # Previous trial Q values
	choice # Chice on current trial
	)
	
	# Initialize new Q value
	Q_new = Q .+ 0 * α # Adding zero times α makes sure that Q_new can accomodate ForwardDiff.Dual

	#  Compute prediction error
	PE = R * ρ  - Q[prev_choice]

	# Update Q
	Q_new[prev_choice] += α * PE
	
	# Compute log probabilities of choosing each stimulus
    log_probs = Q_new .- logsumexp(Q_new)

	# Return log likelihood
	return log_probs[choice]
end

# ╔═╡ cfd9fa84-0b97-45d2-aab9-ae2d236f228f
function Q_learning_trial_FI(
	α, # Learning rate
	ρ, # Reward sensitivity
	R, # Reward on previous trial
	prev_choice, # Choice on previous trial,
	Q, # Previous trial Q values
	choice # Chice on current trial
	) 

	# Function to differentiate
	loglikelihood(θ) = Q_learning_log_likelihood(θ[1], 
		θ[2],
		R,
		prev_choice,
		Q,
		choice
		)

	# Compute Fisher Information as the negative Hessian
	I = -1 .* ForwardDiff.hessian(loglikelihood, [α, ρ])

	# Return the determinant
	return I

end

# ╔═╡ 9c1c571b-ed6d-4e96-bab4-a00a7e23074a
function Q_learning_block_FI(
	sim_dat::AbstractDataFrame
)

	I = zeros(Float64, 2, 2)
    n = nrow(sim_dat)

    # Precompute or preallocate necessary data structures
    α = sim_dat[:, :α]
    ρ = sim_dat[:, :ρ]
    outcome = sim_dat[:, :outcome]
    choice = sim_dat[:, :choice]
    EV_A = sim_dat[:, :EV_A]
    EV_B = sim_dat[:, :EV_B]

	for i in 2:nrow(sim_dat)
		I += Q_learning_trial_FI(
			α[i],
			ρ[i],
			outcome[i-1],
			choice[i-1],
			[EV_A[i-1], EV_B[i-1]],
			choice[i]
		)
	end

	return I

end

function Q_learning_dataset_det_FI(
	sim_dat::DataFrame
)

	# Calculate Fisher Informatio per participant, block
	I = zeros(Float64, 2, 2)

	lk = ReentrantLock()  # Create a lock to synchronize access to I

	grouped_data = groupby(sim_dat, [:PID, :block])
	n_blocks = length(grouped_data)

	Threads.@threads for idx in 1:n_blocks

		block_I = Q_learning_block_FI(grouped_data[idx])

		lock(lk) do
            I += block_I
        end
	end

	return det(I)
end

function Q_learning_simulate_compute_FI(
	n_participants::Int64,
	n_blocks::Int64,
	n_trials::Int64, # Per block
	feedback_magnitudes::Vector{Float64},
	feedback_ns::Vector{Int64}, # Should be same length as feedback_magnitudes and sum to n_trials
	μ_a::Float64, # Means of Φ−1(α), learning rate
	σ_a::Float64, # SDs of Φ−1(α), learning rate
	μ_ρ::Float64, # Means of reward sensitivity
	σ_ρ::Float64; # SDs of reward sensitivity
	random_seed::Int64 = 0,
	feedback_common::Vector{Vector{Int64}} = vcat([[n_trials], [n_trials-1], [n_trials-1], [n_trials-2]],
	repeat([[n_trials-3]], n_blocks - 4)),
	stop_after::Union{Int64, Missing} = missing
)
	sim_dat = simulate_q_learning_dataset(
		n_participants,
		n_trials,
		repeat([feedback_magnitudes], n_blocks),
		repeat([feedback_ns], n_blocks),
		feedback_common,
		μ_a,
		σ_a,
		μ_ρ,
		σ_ρ;
		random_seed = random_seed,
		stop_after = stop_after
	)

	if !ismissing(stop_after)
		filter!(x -> x.choice > 0, sim_dat)
	end

	return Q_learning_dataset_det_FI(sim_dat)

end

@memoize LRU(maxsize=100) function Q_learning_μ_σ_range_FI(
	n_blocks::Int64,
	n_trials::Int64, # Per block
	feedback_magnitudes::Vector{Float64},
	feedback_ns::Vector{Int64}, # Should be same length as feedback_magnitudes and sum to n_trials
	μ_a::AbstractVector, # Grid value for μ_a
	μ_ρ::AbstractVector; # Grid value for μ_ρ
	n_participants::Int64=1,
	σ_a::AbstractVector=[0.], # SD of a, 0. for single participant
	σ_ρ::AbstractVector=[0.], # SD of ρ, 0. for single participant
	random_seeding::String="same", # Whether to use the same random seed for each point or the grid, or different seeds
	feedback_common::Vector{Vector{Int64}} = vcat([[n_trials], [n_trials-1], [n_trials-1], [n_trials-2]],
	repeat([[n_trials-3]], n_blocks - 4)),
	stop_after::Union{Int64, Missing} = missing
)

	# Prepare matrix to hold FI values
	FI = zeros(Float64, length(μ_a), length(μ_ρ), length(σ_a), length(σ_ρ))

	# Compute FI for each parameter combinations
	for (i, μ_ai) in enumerate(μ_a)
		for (j, μ_ρj) in enumerate(μ_ρ)
			for (l, σ_al) in enumerate(σ_a)
				for (m, σ_ρm) in enumerate(σ_ρ)
					FI[i,j,l,m] += Q_learning_simulate_compute_FI(
								n_participants,
								n_blocks,
								n_trials,
								feedback_magnitudes,
								feedback_ns,
								μ_ai,
								σ_al,
								μ_ρj,
								σ_ρm;
								random_seed = random_seeding == "same" ? 0 : 
									parse(Int, 
										string(i) * string(j) * 
										string(l) * string(m)),
								feedback_common = feedback_common,
								stop_after = stop_after
								)
				end
			end
		end
	end

	return dropdims(FI, dims = tuple(findall(size(FI) .== 1)...))

end

function compute_avg_zscored_FI(
	n_trials::Int64,
	n_confusing::Int64;
	stop_after::Union{Int64, Missing} = missing,
	sim_n_blocks::Int64 = 500,
	reward_magnitudes::Vector{Float64} = [1., 2.],
	reward_ns::Vector{Int64} = [floor(Int64, n_trials / 2), ceil(Int64, n_trials / 2)],
	param_grid_res::Int64 = 20,
	μ_a::AbstractVector = range(-2., 1., param_grid_res),
	μ_ρ::AbstractVector = range(0.01, 1.2, param_grid_res),
	random_seeding::String = "different"
)

	# Find all unique arrangements of n confusing trials within N trials
	feedback_commons_idx = combinations(1:n_trials, n_trials - n_confusing)

	feedback_commons = [[i in c ? 
			1 : 0 for i in 1:n_trials] for c in feedback_commons_idx]

	n_total = length(feedback_commons)

	# Simulate
	FI_file = "saved_models/FI/feedback_common_FI_for_experiment_$(n_confusing)_confusing$(!ismissing(stop_after) ? "_stop_after_$stop_after" : "").jld2"
	if !isfile(FI_file)
		FI = [Q_learning_μ_σ_range_FI(sim_n_blocks, n_trials, reward_magnitudes, 
			reward_ns, 
			μ_a, μ_ρ; feedback_common = repeat([fc], sim_n_blocks),
			random_seeding = random_seeding,
			stop_after = stop_after) for fc in feedback_commons]
		@save FI_file FI
	else
		@load FI_file FI
	end

	# zscore by element
	# Initialize arrays to store the mean and standard deviation for each element (i, j)
	mean_FI = zeros(param_grid_res, param_grid_res)
	std_FI = zeros(param_grid_res, param_grid_res)
	
	# Compute the mean and std for each element (i, j) across all matrices
	for i in 1:param_grid_res
	    for j in 1:param_grid_res
	        mean_FI[i, j] = mean([FI[k][i, j] for k in 1:length(FI)])
			std_FI[i, j] = std([FI[k][i, j] for k in 1:length(FI)])
	    end
	end
		
	# Z-score each element (i, j) in all matrices
	zscored_FI = [((FI[k] .- mean_FI) ./ std_FI) for k in 1:length(FI)]

	# Average per each sequence
	avg_FI = [mean(fi) for fi in zscored_FI]


	return feedback_commons, avg_FI

end

# Get best sequences from simulation
function get_sequences_by_FI(
	n_sequences::Int64,
	n_trials::Int64, # Per experimental block
	n_confusing::Int64, # Number of confusing feedback trials in block
	ω_FI::Float64; # Relative weight of FI, vs uniformity of confusing feedback location distirbution in finding sequences
	sim_n_blocks::Int64 = 500, # For FI simulation, governs amount of data and computation time,
	param_grid_res::Int64 = 20, # For FI simulations governs amount of data and computation time,
	random_seeding::String="different", # Whether FI simulations should use same random seed or different for each point on grid
	reward_magnitudes::Vector{Float64} = [1., 2.],
	reward_ns::Vector{Int64} = [floor(Int64, n_trials / 2), ceil(Int64, n_trials / 2)],
	μ_a::AbstractVector = range(-2., 1., param_grid_res),
	μ_ρ::AbstractVector = range(0.01, 1.2, param_grid_res),
	stop_after::Union{Int64, Missing} = missing
)

	# Compute FIs
	feedback_commons, avg_FI = compute_avg_zscored_FI(
		n_trials,
		n_confusing;
		stop_after = stop_after,
		sim_n_blocks = sim_n_blocks,
		reward_magnitudes = reward_magnitudes,
		reward_ns = reward_ns,
		param_grid_res = param_grid_res,
		μ_a = μ_a,
		μ_ρ = μ_ρ,
		random_seeding = random_seeding
	)

	n_total = length(feedback_commons)

	# Transform feedback_commons to 286x13 matrix
	feedback_mat = collect(transpose(hcat(feedback_commons...)))

	# Choose sequences with optimizer
	selected_idx = optimize_FI_distribution(
		n_trials,
		n_confusing,
		n_total,
		n_sequences,
		feedback_mat,
		ω_FI,
		avg_FI
	)

	selected_sequences = feedback_mat[selected_idx,:]

	chosen_FIs = avg_FI[selected_idx]

	chosen_q_FI = mean(mean(chosen_FIs) .> avg_FI)

	chosen_dist = mean(selected_sequences, dims = 1)

	@info "The chosen feedback sequences have a FI value higher than $(chosen_q_FI * 100)% of possible sequences."

	@info "The distribution of common feedbacks across trial number position is $(round.(chosen_dist, digits = 2))"

	return selected_sequences 
end

function optimize_FI_distribution(
	n_trials::Int64,
	n_confusing::Int64,
	n_total::Int64,
	n_sequences::Int64,
	feedback_mat::Matrix{Int64},
	ω_FI::Float64,
	avg_FI::Vector{Float64},
	
)

	# Set a target mean vector (i.e., equal probability for each position)
	target_mean_vector = fill((n_trials - n_confusing) / n_trials, n_trials)

	# Create the optimization model
	model = Model(HiGHS.Optimizer)

	# Decision variables: x[v] is 1 if vector v is selected, 0 otherwise
	@variable(model, x[1:n_total], Bin)

	# Mean vector variables: mu[i] is the mean of selected vectors at position i
	@variable(model, mu[i = 1:n_trials])

	# Constraint: Exactly n_sequences vectors should be selected
	@constraint(model, sum(x) == n_sequences)

	# Constraints to calculate the mean vector
	for i in 1:n_trials
		@constraint(model, mu[i] == sum(feedback_mat[v, i] * x[v] for v in 1:n_total) / n_sequences)
	end

	# Auxiliary variables for absolute deviations
	@variable(model, abs_dev[1:n_trials])

	# Constraints for absolute deviations
	for i in 1:n_trials
		@constraint(model, abs_dev[i] >= mu[i] - target_mean_vector[i])
		@constraint(model, abs_dev[i] >= target_mean_vector[i] - mu[i])
	end


	# Objective: Maximize the total score and minimize the mean vector deviation
	@objective(model, Max, ω_FI * mean(avg_FI[v] * x[v] for v in 1:n_total) - (1-ω_FI) * mean(abs_dev[i] for i in 1:n_trials)
	)

	# Solve the optimization problem
	set_silent(model)
	optimize!(model)

	# Check the status of the solution
	status = termination_status(model)
	if status == MOI.OPTIMAL
		@info "Optimal solution found"
	elseif status == MOI.INFEASIBLE_OR_UNBOUNDED
		@info "Problem infeasible or unbounded"
	else
		@info "Solver terminated with status: $status"
	end

	# Retrieve the solution
	selected_idx = [v for v in 1:n_total if value(x[v]) > 0.5]

	return selected_idx
end

# Get sequences for a mixture of n_confusing blocks
function get_multiple_n_confusing_sequnces(
	n_trials::Int64,
	n_confusing::Vector{Int64},
	n_blocks_total::Int64,
	ω_FI::Float64,
	reward_magnitudes::Vector{Float64},
	reward_ns::Vector{Int64};
	stop_after::Union{Int64, Missing} = missing
	)

	@assert length(filter(x -> x >= n_trials, n_confusing)) == 0 "Values of n_confusing must be smaller than n_trials"

	### Get feedback sequences for each number of confusing feedback block
	feedback_sequences = Matrix{Int64}(undef, n_blocks_total, n_trials)

	# Set sequence for zero confusing
	zero_nc_idx = findall(x -> x == 0, n_confusing)
	feedback_sequences[zero_nc_idx, :] = 
		ones(Int64, (length(zero_nc_idx), n_trials))

	# Find sequences for all other levels
	for nc in filter(x -> x > 0, unique(n_confusing))

		@info "Looking for optimal sequences with $nc confusing trials"

		# Get indices of all blocks with this number of confusing trials
		nc_idx = findall(x -> x == nc, n_confusing)


		# Choose feedback sequcnes
		feedback_sequences[nc_idx, :] = get_sequences_by_FI(
			length(nc_idx),
			n_trials,
			nc,
			ω_FI;
			reward_magnitudes = reward_magnitudes,
			reward_ns = reward_ns,
			stop_after = stop_after
		)
	end
		
	@info "Found optimal solutions for all sequnces. Overall the distribution of common feedback across blocks is $(round.(mean(feedback_sequences, dims = 1), digits = 2))"

	return feedback_sequences
end