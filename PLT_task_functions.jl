function draw_feedback_magnitudes(
	n_trials::Integer, 
	feedback_values::Vector{Vector{Float64}}, 
	feedback_ns::Vector{Vector{Int64}}, 
	)

	# Check inputs
	@assert length(feedback_values) == length(feedback_ns) "Mismatching numbers of blocks indicated by feedback_value and feedback_ns"

	@assert all([length(b) for b in feedback_values] .== 
		[length(b) for b in feedback_ns]) "Mismatching numbers of values in feedback_value and feedback_ns"

	@assert all([sum(b) for b in feedback_ns] .== n_trials) "Number of apperances for each feedback value don't sum to n_trials in all blocks indicated by feedback_ns" 

	# Create feedback_magnitude
	feedback_magnitude = [shuffle(vcat([ones(n) .* 
		feedback_values[i][j] for (j, n) in enumerate(ns)]...))
		for (i, ns) in enumerate(feedback_ns)]

	return feedback_magnitude

end

function draw_feedback_plan(
	n_trials::Integer, 
	feedback_values::Vector{Vector{Float64}}, 
	feedback_ns::Vector{Vector{Int64}}, 
	feedback_common::Vector{Vector{Int64}} 
)
	# Check inputs
	@assert length(feedback_common) == length(feedback_ns) "Mismatching numbers of blocks indicated by feedback_common and feedback_ns"
	
	@assert all([(length(b) == n_trials) | (length(b) == 1) 
		for b in feedback_common]) "Not all elements of feedback_common are exactly n_trials or 1 in legnth"
	

	# Draw feedback magnitudes
	feedback_magnitude = draw_feedback_magnitudes(
		n_trials,
		feedback_values,
		feedback_ns)

	# Make sure no zero EV block - not supported for now
	@assert all([sum(b) != 0 for b in feedback_magnitude] )
		"Zero EV blocks not supported"

	# Draw feedback correctness per trial if needed
	feedback_common = [length(b) > 1 ? b : 
		shuffle(vcat(ones(Int64, b[1]), zeros(Int64, n_trials - b[1]))) 
		for b in feedback_common]

	# Compute domain (loss / gain) per block
	loss_domain = [sum(b) < 0 for b in feedback_magnitude] .+ 0

	# Compute feedback plan for stimulus A - by convention the higher EV stimulus
	stim_A = [feedback_magnitude[i] .* abs.(fc .- loss_domain[i]) 
		for (i, fc) in enumerate(feedback_common)]

	# Same for B, by convention lower EV stimulus
	stim_B = [feedback_magnitude[i] .* abs.(fc .- (1 - loss_domain[i])) 
		for (i, fc) in enumerate(feedback_common)]

	# Return as data frame
	res = DataFrame(
		feedback_A = vcat(stim_A...),
		feedback_B = vcat(stim_B...)
	)

	return res
end

# For now this function supports only two stimuli in a block of 2AFC trials
function task_structure(
	n_trials::Integer, # Number of trials in block
	feedback_values::Vector{Vector{Float64}}, # Values of non-zero feedback on each block. A vector of unique values per block, arbitrarily sized.
	feedback_ns::Vector{Vector{Int64}}, # Number of each non-zero feedback value on each block. A vector of proportions per block, sized the same as feedback_value
	feedback_common::Vector{Vector{Int64}} # Vector with value for each block. Value can be vector n_trials long, with 1 for common and 0 for rare type feedback, or a single number of common type feedbacks for the block. In the latter case, common/rare will be randomly ordered.
	)
	
	n_blocks = length(feedback_values)

	# Set up task structure
	task = DataFrame(
		block = vcat([repeat([i], n_trials) for i in 1:n_blocks]...),
		trial = repeat(1:n_trials, n_blocks)
	)

	# Draw feedback per trial
	task = hcat(task, draw_feedback_plan(n_trials,
		feedback_values,
		feedback_ns,
		feedback_common))

	return task 
end

function softmax_choice(
	logits::Vector{Float64}, # Logit values of each option
	β_1::Float64 # Inverse temperature
	)

	# Compute probabilities
	exp_logits = exp.(logits .* β_1)
    probs = exp_logits ./ sum(exp_logits)

	# Draw chioce
	choice = rand(Categorical(probs))

	return choice
end

function inv_logit(logits::Vector{Float64})
	# Compute probabilities
	exp_logits = exp.(logits .- maximum(logits))  # Subtract max for numerical stability
    return exp_logits ./ sum(exp_logits)
end

function softmax_choice_direct(
	logits::Vector{Float64}, # Logit values of each option
	)

	probs = inv_logit(logits)

	# Draw chioce
	choice = rand(Categorical(probs))

	return choice
end

function q_learning_update(
	R::Float64, # Observed reward
	Q::Float64, # Old Q value
	α::Float64 # Learning rate
	)

	# Prediction error
	PE = R - Q

	# Update Q
	Q_new = Q + α * PE

	return Q_new
end

function q_learning_w_rho_update(
	R::Float64, # Observed reward
	Q::Float64, # Old Q value
	α::Float64, # Learning rate
	ρ::Float64 # Feedback sensitivity
	)

	# Prediction error
	PE = R * ρ  - Q

	# Update Q
	Q_new = Q + α * PE

	return Q_new
end

function policy_gradient_update(
	R::Float64, # Observed reward
	choice::Int64, # Chosen action
	trial::Int64, # Trial number
	H::Vector{Float64}, # Old action propensities
	R_hat::Float64, # Previous timepoint average reward
	α::Float64, # Learning rate
	ρ::Float64 # Feedback sensitivity
)
	# Written based on Sutton and Barto 2018, p. 71

	# Weigthed PE
	wPE = α * (ρ * R - R_hat)

	# Update chosen option propensity
	Π_choice = inv_logit(H)[choice]

	H_new = H .+ wPE .* ((1:length(H) .== choice) .- Π_choice)

	# Update average reward
	R_hat_new = R_hat + wPE #1/trial * (ρ * R - R_hat) this is for the fixed probability case. 
	# But since we want to be able to carry this across blocks, using the drifting probaiblity case

	return H_new, R_hat_new
end

# Helper function to transform indices to upppercase letters
n2l(n::Int64) = Char(n+64)

# Function to simulate trials for single block
function simulate_block(
	task::AbstractDataFrame,
	n_choices::Int64, # Number of stimuli to choose from on each trial
	init::Float64, # Initial values for EVs
	learning_function::Function, # Learning function
	learning_function_arg_cols::Vector{Symbol}, # Columns in task DataFrame containing arguments for learning function other than R and EVs
	choice_function::Function, # Choice function
	chioce_function_arg_cols::Vector{Symbol}; # Columns in task DataFrame containing arguments for choice function other than EVs
	stop_after::Union{Int64, Missing} = missing # Stop after n correct choices. Missing means don't stop
	)


	# Preallocate choice, outcome and EV columns
	n_trials = nrow(task)
	res = DataFrame(
		choice = zeros(Int64, n_trials),
		outcome = zeros(Float64, n_trials)
	)
	
	for c in 1:n_choices
		insertcols!(res, 
			Symbol("EV_$(n2l(c))") =>
			repeat([init], n_trials))
	end

	# Choose, observe outcome and learn for each trial
	for i in 1:n_trials

		# Make choice
		choice_args = vcat([[res[i, Symbol("EV_$(n2l(c))")] for c in 1:n_choices]],
			[task[i, a] for a in chioce_function_arg_cols]) # # Prepare arguements to choice function: EVs and parameters
		res[i, :choice] = choice_function(choice_args...) # Draw choice
		chosen_stim = n2l(res[i, :choice]) # Letter stim

		# Observe feedback
		res[i, :outcome] = R = task[i, Symbol("feedback_$(chosen_stim)")]

		# Update EVs
		if (i < n_trials)
			learning_args = vcat([R, res[i, Symbol("EV_$(chosen_stim)")]],
				[task[i, a] for a in learning_function_arg_cols]) # Prepare arguements to learning function: R, chosen stim EV, and parameters
			res[i + 1, Symbol("EV_$(chosen_stim)")] = 
				learning_function(learning_args...)

			# Carry over values of non-chosen stimuli
			non_chosen = n2l.([x for x in 1:n_choices if x != res[i, :choice]])
			for s in non_chosen
				res[i + 1, Symbol("EV_$(s)")] = 
				res[i, Symbol("EV_$(s)")]
			end
		end

		# Check if stop_after number of optimal chioces were made. Optimal chioce is 1, since optimal stim is A by convention
		if !ismissing(stop_after) && i >= stop_after
			if all(res[(i - stop_after + 1):i, :choice] .== 1)
				break
			end
		end
	end

	return res
end

# Simulate dataset - draw task plan and participant parameters
function simulate_q_learning_dataset(
	n_participants::Int64,
	n_trials::Int64, # Number of trials per block
	feedback_values::Vector{Vector{Float64}}, # Values of non-zero feedback on each block. A vector of unique values per block, arbitrarily sized.
	feedback_ns::Vector{Vector{Int64}}, # Number of each non-zero feedback value on each block. A vector of proportions per block, sized the same as feedback_value
	feedback_common::Vector{Vector{Int64}}, # Vector with value for each block. Value can be vector n_trials long, with 1 for common and 0 for rare type feedback, or a single number of common type feedbacks for the block. In the latter case, common/rare will be randomly ordered.
	μ_a::Float64, # Mean of Φ−1(α), learning rate
	σ_a::Float64, # SD of Φ−1(α), learning rate
	μ_ρ::Float64, # Mean of reward sensitivity
	σ_ρ::Float64; # SD of reward sensitivity
	random_seed::Int64=0, # This is for drawing participants. 
	stop_after::Union{Int64, Missing} = missing, # Stop after n correct choices. Missing means don't stop
	aao::Union{Vector{Float64}, Missing} = missing # Initial value for Q learner
	)

	# Prepare task structure
	Random.seed!(0)
	task = task_structure(n_trials, 
		feedback_values, 
		feedback_ns, 
		feedback_common
	)

	sims = simulate_q_learning_dataset(
				n_participants,
				n_trials,
				task,
				μ_a,
				σ_a,
				μ_ρ,
				σ_ρ;
				random_seed=random_seed,
				stop_after = stop_after,
				aao = aao
	)

	return sims
end

# Simulate dataset from given trial plan, draw participant parameters
function simulate_q_learning_dataset(
	n_participants::Int64,
	task::DataFrame,
	μ_a::Float64, # Mean of Φ−1(α), learning rate
	σ_a::Float64, # SD of Φ−1(α), learning rate
	μ_ρ::Float64, # Mean of reward sensitivity
	σ_ρ::Float64; # SD of reward sensitivity
	random_seed::Int64=0, # This is for drawing participants. 
	stop_after::Union{Int64, Missing} = missing, # Stop after n correct choices. Missing means don't stop
	aao::Union{Vector{Float64}, Missing} = missing # Initial value for Q learner

)
	# Draw participant parameters
	a_dist = Normal(μ_a, σ_a)
	ρ_dist = Normal(μ_ρ, σ_ρ)

	Random.seed!(random_seed)
	
	ρ = rand(ρ_dist, n_participants)
	α = cdf.(Normal(), rand(a_dist, n_participants))

	sims = simulate_q_learning_dataset(
		task,
		α, # Learning rate for each participatn
		ρ; # Reward sensitivity for each participant
		random_seed = missing, # We've already set the random seed. 
		stop_after = stop_after, # Stop after n correct choices. Missing means don't stop
		aao = aao
	)

	return sims
end

# Simulate data set from given participant parameters but draw trial plan
function simulate_q_learning_dataset(
	n_trials::Int64, # Number of trials per block
	feedback_values::Vector{Vector{Float64}}, # Values of non-zero feedback on each block. A vector of unique values per block, arbitrarily sized.
	feedback_ns::Vector{Vector{Int64}}, # Number of each non-zero feedback value on each block. A vector of proportions per block, sized the same as feedback_value
	feedback_common::Vector{Vector{Int64}}, # Vector with value for each block. Value can be vector n_trials long, with 1 for common and 0 for rare type feedback, or a single number of common type feedbacks for the block. In the latter case, common/rare will be randomly ordered.
	α::Vector{Float64}, # Learning rate for each participatn
	ρ::Vector{Float64}; # Reward sensitivity for each participant
	random_seed::Int64=0, # This is for drawing participants. 
	stop_after::Union{Int64, Missing} = missing, # Stop after n correct choices. Missing means don't stop
	aao::Union{Vector{Float64}, Missing} = missing # Initial value for Q learner

)

	# Prepare task structure
	Random.seed!(0)
	task = task_structure(n_trials, 
		feedback_values, 
		feedback_ns, 
		feedback_common
	)

	sims = simulate_q_learning_dataset(
		task,
		α, # Learning rate for each participatn
		ρ, # Reward sensitivity for each participant
		random_seed=random_seed,
		stop_after = stop_after,
		aao = aao
	)

	return sims
end


# Simulate data set from given trial plan and participant parameters
function simulate_q_learning_dataset(
	task::DataFrame,
	α::Vector{Float64}, # Learning rate for each participatn
	ρ::Vector{Float64}; # Reward sensitivity for each participant
	random_seed::Union{Int64, Missing}=missing, 
	stop_after::Union{Int64, Missing} = missing, # Stop after n correct choices. Missing means don't stop
	aao::Union{Vector{Float64}, Missing} = missing # Initial value for Q learner
)
	
	# Prepare DataFrame
	participants = DataFrame(
		PID = 1:length(α),
		ρ = ρ,
		α = α
	)

	# For initial Q values, get the average reward in each block
	if ismissing(aao)
		avg_reward = compute_avg_reward(task)
	else
		avg_reward = aao
	end

	# Combine into single DataFrame
	task = crossjoin(participants, task)

	# Convenience function for simulation
	function simulate_grouped_block(grouped_df)
		simulate_block(grouped_df,
			2,
			avg_reward[grouped_df[1, :block]] .* grouped_df.ρ[1], # Use average reward modulated by rho as initial Q value
			q_learning_w_rho_update,
			[:α, :ρ],
			softmax_choice_direct,
			Vector{Symbol}();
			stop_after = stop_after
		)
	end

	# Set random seed if requested
	if !ismissing(random_seed)
		Random.seed!(random_seed)
	end

	# Simulate data per participants, block
	grouped_task = groupby(task, [:PID, :block])
	sims = transform(grouped_task, simulate_grouped_block)

	return sims
end

# Function to compute average reward in block for initial values
function compute_avg_reward(
	task::AbstractDataFrame
)
	if "valence" in names(task)
		avg_reward = combine(
				groupby(task, :valence),
				[:feedback_A, :feedback_B] => ((a, b) -> mean(vcat(a, b))) => :avg_reward
				)

		blocks = sort(unique(task[!, [:block, :valence]]), :block)

		avg_reward = leftjoin(blocks, avg_reward, on = :valence).avg_reward
	else
		avg_reward = combine(
				groupby(task, :block),
				[:feedback_A, :feedback_B] => ((a, b) -> mean(vcat(a, b))) => :avg_reward
				).avg_reward
	end
	
	return avg_reward
end

function simulate_groups_q_learning_dataset(
	n_participants::Int64, # Number of participants per group
	n_trials::Int64, # Number of trials per block
	μ_a::Vector{Float64}, # Means of Φ−1(α), learning rate
	σ_a::Vector{Float64}, # SDs of Φ−1(α), learning rate
	μ_ρ::Vector{Float64}, # Means of reward sensitivity
	σ_ρ::Vector{Float64}; # SDs of reward sensitivity
	stop_after::Union{Int64, Missing} = missing, # Stop after n correct choices. Missing means don't stop
	feedback_magnitudes::Union{Vector{Vector{Float64}}, Missing} = missing, # Values of non-zero feedback on each block. A vector of unique values per block, arbitrarily sized.
	feedback_ns::Union{Vector{Vector{Int64}}, Missing} = missing, # Number of each non-zero feedback value on each block. A vector of proportions per block, sized the same as feedback_value
	feedback_common::Union{Vector{Vector{Int64}}, Missing} = missing, # Vector with value for each block. Value can be vector n_trials long, with 1 for common and 0 for rare type feedback, or a single number of common type feedbacks for the block. In the latter case, common/rare will be randomly ordered.
	task::Union{DataFrame, Missing} = missing # Alternatively, supply prepared task for simulation
	)

	# Check inputs
	@assert all(length.([μ_a, σ_a, μ_ρ]) .== length(σ_ρ)) "Different number of groups implied by parameter inputs"

	n_groups = length(μ_a)

	# Simulate per group
	sims = []
	for g in 1:n_groups

		if ismissing(task)
			# Draw data
			sim = simulate_q_learning_dataset(
				n_participants,
				n_trials,
				feedback_magnitudes,
				feedback_ns,
				feedback_common,
				μ_a[g],
				σ_a[g],
				μ_ρ[g],
				σ_ρ[g];
				random_seed=g,
				stop_after = stop_after
			)
		else
			# Draw data
			sim = simulate_q_learning_dataset(
				n_participants,
				n_trials,
				task,
				μ_a[g],
				σ_a[g],
				μ_ρ[g],
				σ_ρ[g];
				random_seed=g,
				stop_after = stop_after
			)
		end

		# Add group index
		insertcols!(sim, 2, :group => g)

		# Push to array
		push!(sims, sim)
	end

	# Combine to one DataFrame
	sims = vcat(sims...)

	return sims
end
