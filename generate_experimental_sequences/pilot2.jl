### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 784d74ba-21c7-454e-916e-2c54ed0e6911
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using Random, DataFrames
end

# ╔═╡ 45394a5a-6e96-11ef-27e9-5dddb818f955
"""
Assigns stimulus filenames and determines the optimal stimulus in each pair.

# Arguments
- `n_phases::Int64`: Number of phases (or blocks) in the session.
- `n_pairs::Vector{Int64}`: Vector containing the number of pairs in each block. Assumes the same number of pairs for all phases.
- `categories::Vector{String}`: Vector of category labels to generate stimulus filenames. Default is the combination of letters 'A' to 'Z' and 'a' to 'z', repeated as necessary to cover the number of stimuli required.

# Returns
- `stimulus_A::Vector{String}`: Vector of filenames for the "A" stimuli in each pair.
- `stimulus_B::Vector{String}`: Vector of filenames for the "B" stimuli in each pair.
- `optimal_A::Vector{Int64}`: Vector indicating which stimulus in each pair is the optimal one (1 if stimulus A is optimal, 0 if stimulus B is optimal).

# Description
1. The function first validates that the number of blocks is even and that there are enough categories to cover all stimuli.
2. It generates filenames for two stimuli per pair: `stimulus_A` and `stimulus_B`. The filenames are based on the provided `categories` vector, with "2.png" for `stimulus_A` and "1.png" for `stimulus_B`.
3. The function then randomly assigns which stimulus in each pair is the optimal one (`optimal_A`), ensuring that exactly half of the stimuli are marked as optimal in a balanced way.
4. A loop ensures that the repeating category in each block and the optimal stimulus are relatively independent.

# Constraints
- Assumes an even number of blocks per session.
- Ensures that there are enough category labels to generate filenames for all stimuli in all phases.
"""
function assign_stimuli_and_optimality(;
	n_phases::Int64,
	n_pairs::Vector{Int64}, # Number of pairs in each block. Assume same for all phases
	categories::Vector{String} = [('A':'Z')[div(i - 1, 26) + 1] * ('a':'z')[rem(i - 1, 26)+1] 
		for i in 1:(sum(n_pairs) * 2 * n_phases + n_phases)],
	random_seed::Int64 = 1
	)

	total_n_pairs = sum(n_pairs) # Number of pairs needed
	
	@assert rem(length(n_pairs), 2) == 0 "Code only works for even number of blocks per sesion"

	@assert length(categories) >= sum(total_n_pairs) * n_phases + n_phases "Not enought categories supplied"

	# Compute how many repeating categories we will have
	n_repeating = sum(min.(n_pairs[2:end], n_pairs[1:end-1]))

	# Assign whether repeating is optimal and shuffle
	repeating_optimal = shuffle(
		Xoshiro(random_seed),
		vcat(
			fill(true, div(n_repeating, 2)),
			fill(false, div(n_repeating, 2) + rem(n_repeating, 2))
		)
	)

	# Assign whether categories that cannot repeat are optimal
	rest_optimal = shuffle(
		vcat(
			fill(true, div(total_n_pairs - n_repeating, 2) + 
				rem(total_n_pairs - n_repeating, 2)),
			fill(false, div(total_n_pairs - n_repeating, 2))
		)
	)

	# Initialize vectors for stimuli. A is always novel, B may be repeating
	stimulus_A = []
	stimulus_B = []
	optimal_B = []
	
	for j in 1:n_phases
		for (i, p) in enumerate(n_pairs)
	
			# Choose repeating categories for this block
			n_repeating = ((i > 1) && minimum([p, n_pairs[i - 1]])) * 1
			append!(
				stimulus_B,
				stimulus_A[(end - n_repeating + 1):end]
			)
	
			# Fill up stimulus_repeating with novel categories if not enough to repeat
			for _ in 1:(p - n_repeating)
				push!(
					stimulus_B,
					popfirst!(categories)
				)
			end
			
			# Choose novel categories for this block
			for _ in 1:p
				push!(
					stimulus_A,
					popfirst!(categories)
				)
			end

			# Populate who is optimal vector
			for _ in 1:(n_repeating)
				push!(
					optimal_B,
					popfirst!(repeating_optimal)
				)
			end

			for _ in 1:(p - n_repeating)
				push!(
					optimal_B,
					popfirst!(rest_optimal)
				)
			end
		end
	end

	stimulus_A = (x -> x * "1.png").(stimulus_A)
	stimulus_B = (x -> x * "2.png").(stimulus_B)

	return DataFrame(
		phase = repeat(1:n_phases, inner = total_n_pairs),
		block = repeat(
			vcat([fill(i, p) for (i, p) in enumerate(n_pairs)]...), n_phases),
		pair = repeat(
			vcat([1:p for p in n_pairs]...), n_phases),
		stimulus_A = stimulus_A,
		stimulus_B = stimulus_B,
		optimal_A = .!(optimal_B)
	)

end

# ╔═╡ e658e57d-bbe5-4d15-b70e-b5dedad13d80
assign_stimuli_and_optimality(;
	n_phases = 1,
	n_pairs = [1, 2, 3, 1]
)

# ╔═╡ 94a4ac24-2d30-4410-ae5f-6432f9e2973e
function generate_multiple_n_confusing_sequences(;
		n_trials::Vector{Int64}, # Per block
		n_confusing::Vector{Int64},
		n_pairs::Vector{Int64} # Per block
)

	@assert all(rem.(n_trials, n_pairs) .== 0) "Not all n_trials divisible by n_pairs"

	seqs = []

	for (i, (t, c, p)) in enumerate(zip(n_trials, n_confusing, n_pairs))

		nttp = div(t, p) # Number of trials per pair

		seq = DataFrame(
			cblock = fill(i, t),
			pair = repeat(1:p, inner = nttp),
			appearance = repeat(1:nttp, outer = p),
			feedback_common = vcat(
				[
					shuffle(
						vcat(
							fill(false, c),
							fill(true, nttp - c)
						)
					)
					for _ in 1:p
				]...
			)
		)
		push!(seqs, seq)
	end

	return vcat(seqs...)

end

# ╔═╡ 04caadb8-6095-46e2-84b3-67eb535f4725
function prepare_task_strucutre(;
	n_sessions::Int64,
	n_blocks::Int64,
	n_trials::Vector{Int64}, # Per block
	n_pairs::Vector{Int64}, # Per block
	n_confusing::Vector{Int64}, # Per block
	valence::Vector{Int64}, # Per block
	categories::Vector{String},
	stop_after::Union{Int64, Missing},
	output_file::String,
	high_reward_magnitudes::Vector{Vector{Float64}}, # Per block
	low_reward_magnitudes::Vector{Vector{Float64}} # Per block
) 

	# Checks
	n_blocks_total = n_sessions * n_blocks
	
	@assert length(n_confusing) == n_blocks_total "Length of n_confusing does not match total number of blocks specified"

	@assert length(n_pairs) == n_blocks_total "Length of n_pairs does not match total number of blocks specified"

	@assert length(n_trials) == n_blocks_total "Length of n_trials does not match total number of blocks specified"

	@assert length(valence) == n_blocks_total "Length of valence does not match total number of blocks specified"

	@assert length(high_reward_magnitudes) == n_blocks_total "Length of high_reward_magnitudes does not match total number of blocks specified"

	@assert length(low_reward_magnitudes) == n_blocks_total "Length of low_reward_magnitudes does not match total number of blocks specified"

	# Assign stimulus pairs
	stimuli = assign_stimuli_and_optimality(;
		n_phases = n_sessions,
		n_pairs = n_pairs,
		categories = categories
	)

	rename!(stimuli, :phase => :session) # For compatibility with multi-phase sessions

	# Create sequences of confusing / common feedback
	feedback_sequences = generate_multiple_n_confusing_sequences(
		n_trials = n_trials,
		n_confusing = n_confusing,
		n_pairs = n_pairs
	)

	# Prepare task DataFrame
	task = DataFrame(
		session = repeat(1:n_sessions, inner = sum(n_trials)),
		block = vcat([fill(b, t) for (b, t) in enumerate(n_trials)]...),
		valence = vcat([fill(valence[b], t) for (b, t) in enumerate(n_trials)]...),
		pair = vcat([repeat(1:p, inner = div(n_trials[i], p)) for (i, p) in enumerate(n_pairs)]...)
	)

	# Cumulative block number across sessions
	task.cblock = task.block .+ (task.session .- 1) .* maximum(task.block)

	# Shuffle pair in trial
	transform!(
		groupby(task, [:session, :block, :cblock]),
		:pair => shuffle => :pair
	)

	# Number apperanece of each pair
	transform!(
		groupby(task, [:session, :block, :pair]),
		:pair => (x -> 1:length(x)) => :appearance
	)
	
	# Number trials
	transform!(
		groupby(task, [:session, :block]),
		:pair => (x -> 1:length(x)) => :trial
	)
	
	# Join with sequences	
	task = innerjoin(
		task,
		feedback_sequences,
		on = [:cblock, :pair, :appearance],
		order = :left
	)

	# Join with stimuli
	task = innerjoin(
		task,
		stimuli,
		on = [:session, :block, :pair],
		order = :left
	)

	# Assign whether optimal stimulus is on right
	transform!(
		groupby(task, :cblock),
		:trial => (x -> shuffle(
			collect(Iterators.take(Iterators.cycle(shuffle([true, false])), length(x)))
		)) => :optimal_right
	)

	# Assign stimulus on right
	task.stimulus_right = ifelse.(
		task.optimal_right,
		ifelse.(
			task.optimal_A,
			task.stimulus_A,
			task.stimulus_B
		),
		ifelse.(
			task.optimal_A,
			task.stimulus_B,
			task.stimulus_A
		)
	)

	# Assign stimulus on left
	task.stimulus_left = ifelse.(
		task.optimal_right,
		ifelse.(
			task.optimal_A,
			task.stimulus_B,
			task.stimulus_A
		),
		ifelse.(
			task.optimal_A,
			task.stimulus_A,
			task.stimulus_B
		)
	)

	# Shuffle reward magnitudes
	transform!(
		groupby(task, [:session, :block, :pair]),
		:cblock => (
			x -> shuffle(collect(Iterators.take(
				Iterators.cycle(shuffle(high_reward_magnitudes[x[1]])),
				length(x)
			)))
		) => :high_magnitude,
		:cblock => (
			x -> shuffle(collect(Iterators.take(
				Iterators.cycle(shuffle(low_reward_magnitudes[x[1]])),
				length(x)
			)))
		) => :low_magnitude
	)

	# Compute better feedback value
	task.better_feedback = ifelse.(
		task.valence .== 1,
		task.high_magnitude,
		.- task.low_magnitude
	)

	task.worse_feedback = ifelse.(
		task.valence .== 1,
		task.low_magnitude,
		.- task.high_magnitude
	)

	# Assign feedback to stimulus
	task.feedback_right = ifelse.(
		task.feedback_common,
		ifelse.(
			task.optimal_right,
			task.better_feedback,
			task.worse_feedback
		),
		ifelse.(
			.!task.optimal_right,
			task.better_feedback,
			task.worse_feedback
		)
	)

	task.feedback_left = ifelse.(
		task.feedback_common,
		ifelse.(
			.!task.optimal_right,
			task.better_feedback,
			task.worse_feedback
		),
		ifelse.(
			task.optimal_right,
			task.better_feedback,
			task.worse_feedback
		)
	)

	
	task
end

# ╔═╡ 2c31faf8-8b32-4709-ba3a-43ee9376a3c4
prepare_task_strucutre(;
	n_sessions = 1,
	n_blocks = 18,
	n_trials = repeat([10, 20, 30], 6), # Per block
	categories =  [('A':'Z')[div(i - 1, 26) + 1] * ('a':'z')[rem(i - 1, 26)+1] 
		for i in 1:50],
	valence = repeat([1, -1], 9),
	stop_after = 5,
	output_file = "test",
	n_confusing = vcat([0, 1, 2, 2], fill(3, 18 - 4)), # Per block
	n_pairs = repeat(1:3, 6), # Per block
	high_reward_magnitudes = fill([0.5, 1.], 18), # Per block
	low_reward_magnitudes = fill([0.01], 18) # Per block
) 

# ╔═╡ 7fce878f-92a8-4101-9d58-1c39b12e29d1


# ╔═╡ 800f360f-8ad2-40ba-904d-10874848845b
generate_multiple_n_confusing_sequences(;
	n_trials = [10, 20, 30],
	n_confusing = [0, 1, 2],
	n_pairs = [1, 2, 3]
)

# ╔═╡ Cell order:
# ╠═784d74ba-21c7-454e-916e-2c54ed0e6911
# ╠═45394a5a-6e96-11ef-27e9-5dddb818f955
# ╠═e658e57d-bbe5-4d15-b70e-b5dedad13d80
# ╠═04caadb8-6095-46e2-84b3-67eb535f4725
# ╠═2c31faf8-8b32-4709-ba3a-43ee9376a3c4
# ╠═94a4ac24-2d30-4410-ae5f-6432f9e2973e
# ╠═7fce878f-92a8-4101-9d58-1c39b12e29d1
# ╠═800f360f-8ad2-40ba-904d-10874848845b
