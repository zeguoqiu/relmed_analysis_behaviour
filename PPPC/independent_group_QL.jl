### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ fe776a2c-589a-11ef-0331-472f2769c12f
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, ParetoSmooth, MCMCDiagnosticTools, Printf
	using LogExpFunctions: logistic, logit

	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/independent_group_QL.jl")
end

# ╔═╡ ba3495da-93cc-4002-a2a5-99589a819a94
begin
	n_participants = 10
	
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

	block = fill(task.block, n_participants)
	valence = fill(unique(task[!, [:block, :valence]]).valence, n_participants)
	outcomes = fill(outcomes, n_participants)

	prior_model = independent_group_QL(
		block = block,
		valence = valence,
		choice = [fill(missing, length(block[p])) for p in eachindex(block)],
		outcomes = outcomes,
		initV = fill(aao, 1, 2)
	)


	# Draw parameters and simulate choice
	prior_sample = sample(
		prior_model,
		Prior(),
		9
	)

end

# ╔═╡ 4b1235b9-5fa8-4323-acc5-0fcb57a0fc25
begin
    ρ = single_layer_chain_to_vector(prior_sample, "ρ")
	a = single_layer_chain_to_vector(prior_sample, "a")
end

# ╔═╡ 79aaa431-8d0b-4fad-87c1-9d2cc9040641
	# Compute Q values
	generated_quantities(prior_model, prior_sample) |> vec


# ╔═╡ a383c231-79f1-4cde-9e09-8ff46e0b835c
Qs

# ╔═╡ c14b4f42-e328-4ba4-85a7-be65404f1cba
begin
	param = "choice"
	matching_columns = filter(x -> occursin(Regex("$(param)\\[\\d+\\]\\[\\d+\\]"), x), string.(names(prior_sample, :parameters)))
    wide_df = prior_sample[:, matching_columns, 1] |> DataFrame

	select!(wide_df, Not(:chain))
	long_df = stack(wide_df, matching_columns, :iteration, value_name = :choice)

	long_df.PID = (x -> parse(Int64, match(Regex("$(param)\\[(\\d+)\\]\\[(\\d+)\\]"), x)[1])).(long_df.variable)

	long_df.trial = (x -> parse(Int64, 
        match(Regex("$(param)\\[(\\d+)\\]\\[(\\d+)\\]"), x)[2])).(long_df.variable)

	select!(long_df, [:iteration, :PID, :trial, :choice])
	
	long_df
end

# ╔═╡ 96102f32-1bdc-4d11-992c-f003a24fac6a
rand(BernoulliLogit()) + 0

# ╔═╡ Cell order:
# ╠═fe776a2c-589a-11ef-0331-472f2769c12f
# ╠═ba3495da-93cc-4002-a2a5-99589a819a94
# ╠═4b1235b9-5fa8-4323-acc5-0fcb57a0fc25
# ╠═79aaa431-8d0b-4fad-87c1-9d2cc9040641
# ╠═a383c231-79f1-4cde-9e09-8ff46e0b835c
# ╠═c14b4f42-e328-4ba4-85a7-be65404f1cba
# ╠═96102f32-1bdc-4d11-992c-f003a24fac6a
