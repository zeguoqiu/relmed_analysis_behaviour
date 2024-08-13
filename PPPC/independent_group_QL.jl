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
prior_sample = let n_participants = 10
	
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

	# Arrange values for simulation
	block = fill(task.block, n_participants)
	valence = fill(unique(task[!, [:block, :valence]]).valence, n_participants)
	outcomes = fill(outcomes, n_participants)

	# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	simulate_independent_group_QL(
		1;
		block = block,
		valence = valence,
		outcomes = outcomes,
		initV = fill(aao, 1, 2),
		random_seed = 0
	)

end

# ╔═╡ Cell order:
# ╠═fe776a2c-589a-11ef-0331-472f2769c12f
# ╠═ba3495da-93cc-4002-a2a5-99589a819a94
