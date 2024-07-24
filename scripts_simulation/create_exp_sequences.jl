### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 9474a728-2e76-11ef-1e8d-f93766744325
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase,
		ForwardDiff, LinearAlgebra, Memoization, LRUCache, GLM, JLD2, FileIO, JuMP, Cbc, HiGHS, JSON, CSV
	using IterTools: product
	using LogExpFunctions: logsumexp
	using Combinatorics: combinations
	include("task_functions.jl")
	include("fisher_information_functions.jl")
	include("plotting_functions.jl")
end

# ╔═╡ dead4e81-8ec4-4f9b-803f-a5580f15f0a7
function assign_stimuli_and_valence(
	n_phases::Int64,
	n_blocks::Int64; # Per phase
	categories::Vector{String} = [('A':'Z')[div(i - 1, 26) + 1] * ('a':'z')[rem(i - 1, 26)+1] 
		for i in 1:(n_blocks * n_phases + n_phases)]
	)
	
	@assert rem(n_blocks, 2) == 0 "Code only works for even number of blocks per sesion"

	# Compile stimuli filenames
	stimulus_A = [categories[(p - 1) * (n_blocks + 1) + b] * "2.png" 
	  for p in 1:n_phases for b in 1:n_blocks]

	stimulus_B = [categories[(p - 1) * (n_blocks + 1) + b + 1] * "1.png" 
	  for p in 1:n_phases for b in 1:n_blocks]

	not_done = true

	# Decide which is valenced stimulus
	optimal_A = []
	while not_done
		optimal_A = []
		
		for s in 1:n_phases
			push!(optimal_A, 
				shuffle(
					vcat(
						ones(Int64, div(n_blocks, 2)), 
						zeros(Int64, div(n_blocks, 2))
					)
				)
			)
		end

		# Check that we didn't get freak values
		n_matches = [sum(p[1:(end-1)] .== 
			1 .- p[2:end]) for p in optimal_A]

		not_done = any(abs.(n_matches .- div(n_blocks, 2)) .>= (n_blocks / 4))
	end

	return stimulus_A, stimulus_B, vcat(optimal_A...)
end

# ╔═╡ f956676d-bbd2-4cba-8b98-56c7f3b5758a
begin
	n_blocks = 24
	n_sessions = 2
	a = repeat(shuffle(repeat([[0.5, 1.], [1.]], div(n_blocks, 4))), 2 * n_sessions)
	length(a)
end

# ╔═╡ c458b38c-f9f8-4f7c-8af2-cadd270b9eaa
function save_to_JSON(
	df::DataFrame, 
	file_path::String
)
	# Initialize an empty dictionary to store the grouped data
	json_groups = []
	
	# Iterate through unique blocks and their respective rows
	for s in unique(df.session)
		session_groups = []
		for b in unique(df.block)
		    # Filter the rows corresponding to the current block
		    block_group = df[(df.block .== b) .&& (df.session .== s), :]
		    
		    # Convert each row in the block group to a dictionary and collect them into a list
		    push!(session_groups, [Dict(pairs(row)) for row in eachrow(block_group)])
		end
		push!(json_groups, session_groups)
	end
	
	# Convert to JSON String
	json_string = JSON.json(json_groups)
		
	# Write the JSON string to the file
	open(file_path, "w") do file
	    write(file, json_string)
	end

end

# ╔═╡ afecd33f-4df6-4e6d-a426-a4b0b3980364
# Create sequence function
function prepare_task_strucutre(;
	n_sessions::Int64,
	n_blocks::Int64,
	n_trials::Int64,
	categories::Vector{String},
	valence::Vector{Float64},
	valence_grouped::Bool,
	stop_after::Union{Int64, Missing},
	output_file::String,
	ω_FI::Float64 = 0.35,
	n_confusing::Vector{Int64} = repeat(vcat([0 ,1, 1, 2], fill(3, n_blocks - 4)), n_sessions),
	high_reward_magnitudes::Vector{Vector{Float64}} = repeat([[0.5, 1.]], n_blocks * n_sessions),
	low_reward_magnitudes::Vector{Vector{Float64}} = repeat([[0.01]], n_blocks * n_sessions)
) 
	# Checks
	n_blocks_total = n_sessions * n_blocks
	
	@assert length(categories) >= n_blocks_total + n_sessions + n_sessions * valence_grouped "Length of stimuli_filenames is $(length(categories)), while $(n_blocks_total + n_sessions) is needed"

	@assert length(n_confusing) == n_blocks_total "Length of n_confusing does not match total number of blocks specified"

	@assert length(valence) == n_blocks_total "Length of valence does not match total number of blocks specified"

	# Get feedback sequences
	feedback_sequences = get_multiple_n_confusing_sequnces(
		n_trials,
		n_confusing,
		n_blocks_total,
		ω_FI,
		[0.5, 1],
		[div(n_trials, 2), ceil(Int64, n_trials / 2)];
		stop_after = stop_after
	) |> transpose |> collect

	# Compute stimulus n, and shuffle
	Random.seed!(0)
	high_reward_ns = [shuffle([div(n_trials, length(b)) + (i <= n_trials % length(b) ? 1 : 0) for i in 1:length(b)]) for b in high_reward_magnitudes]

	low_reward_ns = [shuffle([div(n_trials, length(b)) + (i <= n_trials % length(b) ? 1 : 0) for i in 1:length(b)]) for b in low_reward_magnitudes]
	
	# Create shuffled magnitude sequences
	high_magnitude_sequences = hcat([valence[b] .* 
		shuffle(vcat([ones(Float64, n) * high_reward_magnitudes[b][i] for (i, n) in enumerate(high_reward_ns[b])]...)) 
			for b in 1:size(feedback_sequences, 2)]...)

	low_magnitude_sequences = hcat([valence[b] .* 
		shuffle(vcat([ones(Float64, n) * low_reward_magnitudes[b][i] for (i, n) in enumerate(low_reward_ns[b])]...)) 
			for b in 1:size(feedback_sequences, 2)]...)

	# Arrange by feedback sequence to get outcomes per stimulus
	high_stim_sequences = ifelse.(feedback_sequences .== 1, 
		high_magnitude_sequences, low_magnitude_sequences) 
	low_stim_sequences = ifelse.(feedback_sequences .== 0, 
		high_magnitude_sequences, low_magnitude_sequences)
	
	# Assign stimuli and valence
	Random.seed!(0)
	stimuli_A, stimuli_B, optimal_A = assign_stimuli_and_valence(
		n_sessions * (valence_grouped + 1),
		div(n_blocks, (valence_grouped + 1));
		categories = categories
		)

	@assert length(stimuli_A) == n_blocks_total

	# Start building task strutcure - draw left / right
	n_trials_total = n_blocks_total * n_trials

	# Helper variable to assign stimulis to right and left an equal number of times
	more_right = repeat([0, 1], div(n_blocks_total, 2)) |> shuffle
	
	# Session, block, trial counters, and assign which stimulus on right
	task = DataFrame(
		session = vcat([fill(i, n_blocks * n_trials) for i in 1:n_sessions]...),
		block = repeat(vcat([fill(i, n_trials) for i in 1:n_blocks]...), n_sessions),
		trial = repeat(1:n_trials, n_blocks_total),
		stimulusA_on_right = vcat([shuffle(vcat(ones(Int64, div(n_trials, 2) + more_right[i]), zeros(Int64, div(n_trials, 2) + (1 - more_right[i])))) for i in 1:n_blocks_total]...),
		stimulus_A = vcat([fill(stimuli_A[i], n_trials) for i in 
			1:n_blocks_total]...),
		stimulus_B = vcat([fill(stimuli_B[i], n_trials) for i in 
			1:n_blocks_total]...),
		optimal_A = vcat([fill(optimal_A[i], n_trials) for i in 	
			1:n_blocks_total]...),
		valence = vcat([fill(valence[i], n_trials) for i in
			1:n_blocks_total]...)
	)

	# Assign stimulus name
	task.stimulus_right = ifelse.(task.stimulusA_on_right .== 1, 
		task.stimulus_A, task.stimulus_B)

	task.stimulus_left = ifelse.(task.stimulusA_on_right .== 0, 
		task.stimulus_A, task.stimulus_B)

	# Assign feedback per stimulus
	task.feedback_A = ifelse.(((task.optimal_A .== 1) .& (task.valence .== 1.)) .|
		(task.optimal_A .== 0) .& (task.valence .== -1.), 
		vec(high_stim_sequences), vec(low_stim_sequences))

	task.feedback_B = ifelse.(((task.optimal_A .== 1) .& (task.valence .== 1.)) .|
		(task.optimal_A .== 0) .& (task.valence .== -1.), 
		vec(low_stim_sequences), vec(high_stim_sequences))

	# Assign feedback per side
	task.feedback_right = ifelse.(task.stimulusA_on_right .== 1, 
		task.feedback_A, task.feedback_B)

	task.feedback_left = ifelse.(task.stimulusA_on_right .== 0, 
		task.feedback_A, task.feedback_B)

	# Compute optimal stimulus
	task.optimal_right = ifelse.(task.optimal_A .== task.stimulusA_on_right, 
		1, 0)

	# Tests
	test_feedback_consistency(x) = abs(sum(sign.(x))) == length(x)

	@assert combine(groupby(task, [:session, :block]),
		:feedback_A => test_feedback_consistency => :test_A,
		:feedback_B => test_feedback_consistency => :test_B
	) |> df -> sum(df.test_A) + sum(df.test_B) == nrow(df) * 2 "Mixed feedback valence blocks found"

	

	@assert mean(task.optimal_A) == 0.5 "Proportion of optimal_A is $(mean(task.optimal_A))"

	# Save to file
	save_to_JSON(task, "results/$output_file.json")
	CSV.write("results/$output_file.csv", task)

	return task

end

# ╔═╡ be38e893-1554-4530-97e7-8656d9198336
function prepare_task_pilot1_by_condition(
	condition::String
)

	# Parse condition
	stop_after = condition[1] == '1' ? 5 : missing
	valence_grouped = condition[2] == '1'
	
	@assert !valence_grouped || length(condition) == 3 "Condition indicates that valence should be grouped, but ordering not given" # Check condition input
	
	reward_first = length(condition) == 3 ? condition[3] == '1' : missing
	
	# Numbers
	n_sessions = 2
	n_blocks = 24
	n_trials = 13

	# Stimuli
	Random.seed!(0)
	categories = shuffle(unique([replace(s, ".png" => "")[1:(end-1)] for s in  readlines("allimages.txt")]))
	

	@assert length(categories) >= n_sessions * n_blocks + n_sessions + 2 "categories needs to be of length $(n_sessions * n_blocks + n_sessions + 2) but it is of length $(length(categories))"

	# Valence
	if valence_grouped
		if reward_first
			valence = vcat([vcat(
				fill(isodd(i) ? 1. : -1., div(n_blocks, 2)),
				fill(isodd(i) ? -1. : 1., div(n_blocks, 2))
			) for i in 1:n_sessions]...)
		else
			valence = vcat([vcat(
				fill(isodd(i) ? -1. : 1., div(n_blocks, 2)),
				fill(isodd(i) ? 1. : -1., div(n_blocks, 2))
			) for i in 1:n_sessions]...)
		end
	else
		valence = vcat([shuffle(vcat(
			fill(1., div(n_blocks, 2)),
			fill(-1., div(n_blocks, 2)))
		) for _ in 1:n_sessions]...)

		@assert sum(valence[1:div(end, n_sessions)]) == 
			sum(valence[(div(end, n_sessions) + 1):end])
	end

	@assert mean(valence) == 0.

	# Reward magnitudes - shuffle arrangement, and then alot by valence, to make sure orthogonal
	high_reward_magnitudes = Vector{Vector{Float64}}(undef, n_blocks * n_sessions)
	high_reward_magnitudes[sortperm(valence)] = repeat(shuffle(repeat([[0.5, 1.], [1.]], div(n_blocks, 4))), 2 * n_sessions)

	low_reward_magnitudes = [length(b) == 2 ? [0.01] : [0.01, 0.5] for b in high_reward_magnitudes]

	@assert [maximum(r) for r in high_reward_magnitudes][sortperm(valence)] |> 
		x -> mean(x[1:div(end, 2)]) == mean(x[(div(end, 2) + 1):end])

	@assert [maximum(r) for r in low_reward_magnitudes][sortperm(valence)] |> 
		x -> mean(x[1:div(end, 2)]) == mean(x[(div(end, 2) + 1):end])


	output_file = "PLT_task_structure_$condition"

	# Prepare structure
	return prepare_task_strucutre(;
		n_sessions = n_sessions,
		n_blocks = n_blocks,
		n_trials = n_trials,
		categories = categories,
		valence = valence,
		valence_grouped = valence_grouped,
		stop_after = stop_after,
		output_file = output_file,
		ω_FI = 0.53,
		high_reward_magnitudes = high_reward_magnitudes,
		low_reward_magnitudes = low_reward_magnitudes
	)

end

# ╔═╡ ac6f83b9-caf9-4a92-950e-dd9300bd4b1f
# Compute and save per condition. Needed function below
let conditions = ["10", "00", "110", "111", "010", "011"]
	for c in conditions
		println(c)
		prepare_task_pilot1_by_condition(c)
	end
end

# ╔═╡ Cell order:
# ╠═9474a728-2e76-11ef-1e8d-f93766744325
# ╠═ac6f83b9-caf9-4a92-950e-dd9300bd4b1f
# ╠═dead4e81-8ec4-4f9b-803f-a5580f15f0a7
# ╠═f956676d-bbd2-4cba-8b98-56c7f3b5758a
# ╠═be38e893-1554-4530-97e7-8656d9198336
# ╠═afecd33f-4df6-4e6d-a426-a4b0b3980364
# ╠═c458b38c-f9f8-4f7c-8af2-cadd270b9eaa
